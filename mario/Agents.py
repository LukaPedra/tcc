import torch
from torch import nn
import random
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class MarioDQNAgent:
    def __init__(self, input_dims, num_actions, learning_rate=0.00025, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_frames=30000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_frames

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN_CNN(input_dims, num_actions).to(self.device)
        self.target_net = DQN_CNN(input_dims, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        self.batch_size = 32

    def get_action(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state.__array__(), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(TensorDict({
            "state": torch.tensor(state.__array__()),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(next_state.__array__()),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size).to(self.device)
        state, action, reward, next_state, done = (batch.get(key) for key in ("state", "action", "reward", "next_state", "done"))

        # FIX: Reshape the tensors to remove the extra dimension
        if state.dim() == 5:
            state = state.squeeze(-1)
        if next_state.dim() == 5:
            next_state = next_state.squeeze(-1)

        # Ensure action is the correct shape
        action = action.unsqueeze(-1)

        current_q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.target_net(next_state).max(1)[0].detach()

        target_q_values = reward + (1 - done.float()) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)

class DQN_CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.flatten(o, 1).size(1))

    def forward(self, x):
        conv_out = self.conv(x / 255.0)
        return self.fc(torch.flatten(conv_out, 1))
