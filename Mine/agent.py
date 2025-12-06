import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = kwargs.get('exploration_rate', 1)
        self.exploration_rate_decay = kwargs.get('exploration_rate_decay', 0.99995)
        self.exploration_rate_min = kwargs.get('exploration_rate_min', 0.1)
        self.gamma = kwargs.get('gamma', 0.9)
        self.burnin = kwargs.get('burnin', 1e5)
        self.learn_every = kwargs.get('learn_every', 3)
        self.sync_every = kwargs.get('sync_every', 1e4)
        self.save_every = kwargs.get('save_every', 5e5)
        self.save_dir = save_dir

        self.curr_step = 0

        # Device Detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders) acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)

        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        Given a state (or batch of states), choose an epsilon-greedy action.
        """
        # Ensure state is a numpy array (handles LazyFrames from Gym)
        state_np = np.array(state)

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            # If state is batched (4 dims: Batch, C, H, W), return random actions for EACH env
            if state_np.ndim == 4:
                action_idx = np.random.randint(self.action_dim, size=state_np.shape[0])
            else:
                action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.tensor(state_np, dtype=torch.float32).to(self.device)

            # Add batch dimension if single input
            if state.ndim == 3:
                state = state.unsqueeze(0)

            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1)

            action_idx = action_idx.cpu().numpy()

            # If input was single (ndim=3), return single int
            if state_np.ndim == 3:
                action_idx = action_idx[0]

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer).
        Handles both single values and batches.
        """
        state = np.array(state)
        next_state = np.array(next_state)

        # Check if input is a batch (Vector Env)
        if state.ndim == 4:
            num_envs = state.shape[0]
            # Loop through the batch and save each experience individually
            for i in range(num_envs):
                s = torch.tensor(state[i], dtype=torch.float32).to(self.device)
                ns = torch.tensor(next_state[i], dtype=torch.float32).to(self.device)
                a = torch.tensor([action[i]], dtype=torch.long).to(self.device)
                r = torch.tensor([reward[i]], dtype=torch.float32).to(self.device)
                d = torch.tensor([done[i]], dtype=torch.bool).to(self.device)
                self.memory.append((s, ns, a, r, d))
        else:
            # Single env case
            s = torch.tensor(state, dtype=torch.float32).to(self.device)
            ns = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            a = torch.tensor([action], dtype=torch.long).to(self.device)
            r = torch.tensor([reward], dtype=torch.float32).to(self.device)
            d = torch.tensor([done], dtype=torch.bool).to(self.device)
            self.memory.append((s, ns, a, r, d))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate