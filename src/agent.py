import torch
import numpy as np
import random
from pathlib import Path
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from neural import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5

        # Memory
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        """
        Given a state (or batch), choose an epsilon-greedy action.
        """
        # Ensure state is tensor on device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)

        is_batch = (state.ndim == 4)

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            if is_batch:
                action_idx = np.random.randint(self.action_dim, size=state.shape[0])
            else:
                action_idx = np.random.randint(self.action_dim)
        # EXPLOIT
        else:
            if not is_batch:
                state = state.unsqueeze(0)

            # Cast to float and normalize here for the network
            state = state.to(dtype=torch.float32).div(255.0)

            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1)
            action_idx = action_idx.cpu().numpy()

            if not is_batch:
                action_idx = action_idx[0]

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        CRITICAL FIX: Enforce Float32/Uint8 to avoid MPS crashes.
        """
        def to_tensor(x, dtype):
            if isinstance(x, torch.Tensor):
                t = x.cpu()
            else:
                t = torch.tensor(x)
            # Enforce the specific dtype (MPS doesn't like Float64)
            return t.to(dtype=dtype)

        # 1. Images -> Uint8 (Saves RAM, MPS Safe)
        state = to_tensor(state, torch.uint8)
        next_state = to_tensor(next_state, torch.uint8)

        # 2. Actions -> Long (Int64 is fine for MPS indices)
        action = to_tensor(action, torch.long)

        # 3. Rewards -> Float32 (CRITICAL FIX: Default was Float64 which crashes MPS)
        reward = to_tensor(reward, torch.float32)

        # 4. Done -> Bool
        done = to_tensor(done, torch.bool)

        if state.ndim == 4:
            if action.ndim == 1: action = action.unsqueeze(1)
            if reward.ndim == 1: reward = reward.unsqueeze(1)
            if done.ndim == 1: done = done.unsqueeze(1)

            td = TensorDict({
                "state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done
            }, batch_size=[state.shape[0]])

            self.memory.extend(td)
        else:
            self.memory.add(TensorDict({
                "state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done
            }, batch_size=[]))

    def recall(self):
        # Sample moves to device. Since we ensured Float32 in cache(), this won't crash now.
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # Neural Net needs Float32 normalized 0-1
        state = state.to(dtype=torch.float32).div(255.0)

        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Neural Net needs Float32 normalized 0-1
        next_state = next_state.to(dtype=torch.float32).div(255.0)

        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

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