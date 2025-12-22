import torch
from torch import nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Backbone partilhado (CNN)
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        # Cabeça do Actor (Policy) -> Probabilidade das ações
        self.actor = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # Cabeça do Critic (Value) -> Valor escalar do estado
        self.critic = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, state):
        features = self.feature_layer(state)

        # Actor
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        # Critic
        value = self.critic(features)

        return dist, value