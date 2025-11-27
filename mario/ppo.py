import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCriticCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        # conv body similar to DQN
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_out_size = int(torch.flatten(conv_out, 1).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        conv_out = self.conv(x)
        flat = torch.flatten(conv_out, 1)
        feat = self.fc(flat)
        return self.policy_head(feat), self.value_head(feat)


class PPOAgent:
    """
    A compact PPO implementation for discrete action spaces.

    Usage:
      agent = PPOAgent(obs_shape, n_actions)
      for step: action, logp, value = agent.act(state)
      store reward, done
      after rollout: agent.update()
    """

    def __init__(
        self,
        obs_shape,
        n_actions,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.net = ActorCriticCNN(obs_shape, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # hyperparams
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # rollout buffer
        self.clear_buffer()

    def clear_buffer(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def act(self, obs):
        # obs: numpy array or torch tensor with shape (C,H,W) or (H,W,C,1) variants
        obs_t = self._to_tensor(obs)

        logits, value = self.net(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)

        return action.item(), logp.item(), value.item()

    def store(self, obs, action, logp, value, reward, done):
        self.obs.append(np.array(obs, copy=True))
        self.actions.append(action)
        self.logprobs.append(logp)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_gae(self, last_value=0.0):
        # convert to tensors
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        returns = np.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns[step] = gae + values[step]

        advantages = returns - values[:-1]
        return returns, advantages

    def update(self, last_obs=None):
        # If rollout ended due to timeout we need last value estimate
        if last_obs is not None:
            with torch.no_grad():
                # use the same observation normalization used in act()
                last_obs_t = self._to_tensor(last_obs)
                _, last_value = self.net(last_obs_t)
                last_value = last_value.item()
        else:
            last_value = 0.0

        returns, advantages = self._compute_gae(last_value=last_value)

        # to tensors (handle possible trailing singleton channel dim from wrappers)
        obs_np = np.array(self.obs)
        if obs_np.ndim == 5 and obs_np.shape[-1] == 1:
            obs_np = obs_np.squeeze(-1)
        obs = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # prepare indices for minibatches
        n_steps = obs.shape[0]
        inds = np.arange(n_steps)

        # accumulators for logging
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_batches = 0

        for epoch in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, n_steps, self.batch_size):
                mb_inds = inds[start : start + self.batch_size]
                mb_obs = obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_logp = old_logprobs[mb_inds]
                mb_returns = returns_t[mb_inds]
                mb_adv = advantages_t[mb_inds]

                logits, values = self.net(mb_obs)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                mb_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # accumulate
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_batches += 1

        # clear buffer
        self.clear_buffer()
        # return averaged metrics for logging
        if total_batches > 0:
            return {
                "actor_loss": total_actor_loss / total_batches,
                "value_loss": total_value_loss / total_batches,
                "entropy": total_entropy / total_batches,
            }
        else:
            return {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    def _to_tensor(self, obs):
        """Convert observation (numpy or torch) to a batched torch tensor (1,C,H,W).
        Handles common wrapper shapes like (C,H,W), (H,W,C) and (C,H,W,1) produced by
        some Gym wrappers when keep_dim is True/False.
        """
        # If it's already a torch tensor
        if isinstance(obs, torch.Tensor):
            t = obs
        else:
            arr = np.array(obs)
            # squeeze trailing singleton channel
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            # If shape is (H,W,C) (channels last), move to (C,H,W)
            if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4) and arr.shape[-1] in (1, 3, 4):
                arr = np.moveaxis(arr, -1, 0)
            # If we somehow have (H,W) make it (1,H,W)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 0)
            t = torch.tensor(arr, dtype=torch.float32)

        # ensure batched
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path, map_location=None):
        self.net.load_state_dict(torch.load(path, map_location=map_location))
