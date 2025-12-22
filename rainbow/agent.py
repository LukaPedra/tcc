import torch
import numpy as np
from collections import deque
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, PrioritizedSampler
from model import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        # Initialize Neural Network
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        self.target_net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.curr_step = 0
        self.save_every = 5e5

        # Learning parameters
        self.gamma = 0.9
        self.n_step = 3  # N-step learning
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

        # PER parameters
        self.beta = 0.4
        self.beta_increment = 0.00001

        # Memory Setup
        sampler = PrioritizedSampler(max_capacity=100000, alpha=0.6, beta=self.beta)
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu")),
            sampler=sampler,
            batch_size=32
        )
        self.batch_size = 32

    def act(self, state):
        # Burnin com ações aleatórias
        if self.curr_step < self.burnin:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)

            self.net.reset_noise()
            with torch.no_grad():
                action_values = self.net(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Armazena experiência usando N-step buffer"""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        # Adiciona ao buffer temporário
        transition = (state, next_state, action, reward, done)
        self.n_step_buffer.append(transition)

        # Se buffer cheio, processa o passo n
        if len(self.n_step_buffer) == self.n_step:
            s, ns, a, r, d = self._get_n_step_info()
            self._add_to_memory(s, ns, a, r, d)

        # Se o episódio acabou, esvazia o buffer
        if done:
            while len(self.n_step_buffer) > 0:
                s, ns, a, r, d = self._get_n_step_info()
                self._add_to_memory(s, ns, a, r, d)
                self.n_step_buffer.popleft()

    def _get_n_step_info(self):
        reward_n = 0
        gamma = 1

        state_0, _, action_0, _, _ = self.n_step_buffer[0]
        _, next_state_n, _, _, done_n = self.n_step_buffer[-1]

        for _, _, _, r, d in self.n_step_buffer:
            reward_n += r * gamma
            if d:
                done_n = True
                break
            gamma *= self.gamma

        return state_0, next_state_n, action_0, reward_n, done_n

    def _add_to_memory(self, state, next_state, action, reward, done):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward]).float()
        done = torch.tensor([done])

        self.memory.add(TensorDict({
            "state": state, "next_state": next_state, "action": action, "reward": reward, "done": done
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        weights = batch.get("_weight")
        indices = batch.get("index")
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), weights, indices

    def td_estimate(self, state, action):
        return self.net(state)[np.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_net(next_state)[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * (self.gamma ** self.n_step) * next_Q).float()

    def update_Q_online(self, td_estimate, td_target, weights):
        elementwise_loss = self.loss_fn(td_estimate, td_target)
        loss = (elementwise_loss * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_Q_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), steps=self.curr_step), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0: self.sync_Q_target()
        if self.curr_step % self.save_every == 0: self.save()
        if self.curr_step < self.burnin: return None, None
        if self.curr_step % self.learn_every != 0: return None, None

        state, next_state, action, reward, done, weights, indices = self.recall()

        # Beta annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        if hasattr(self.memory.sampler, "beta"): self.memory.sampler.beta = self.beta

        # Reset noise for training
        self.net.reset_noise()
        self.target_net.reset_noise()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt, weights)

        # Update priorities
        new_priorities = torch.abs(td_tgt - td_est) + 1e-6
        self.memory.update_priority(indices, new_priorities.detach())

        return (td_est.mean().item(), loss)