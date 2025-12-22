import torch
import numpy as np
from torch.distributions import Categorical
from model import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        # Rede Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=2.5e-4, eps=1e-5)

        # Hiperparâmetros PPO
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.batch_size = 64
        self.n_epochs = 10

        # Buffer para armazenar trajetórias (limpo a cada update)
        self.buffer = []
        self.buffer_size = 2048 # Número de passos antes de atualizar
        self.curr_step = 0

    def act(self, state):
        # 1. Tratar Tupla (State, Info) retornada pelo gym new_step_api
        if isinstance(state, tuple):
            state = state[0]

        # 2. Converter LazyFrames para NumPy Array
        # O Wrapper FrameStack retorna LazyFrames, que o torch.tensor não entende diretamente
        state = np.array(state)

        # 3. Converter para Tensor e adicionar dimensão de Batch
        state = torch.tensor(state, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def remember(self, state, action, log_prob, val, reward, done):
        """Guarda a transição no buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'val': val,
            'reward': reward,
            'done': done
        })

    def learn(self):
        # Só aprende se o buffer estiver cheio
        if len(self.buffer) < self.buffer_size:
            return None, None

        # Converter buffer para listas
        states = np.array([t['state'] for t in self.buffer])
        # Ajuste de dimensões para imagem (Batch, C, H, W) se necessário
        # LazyFrames/Numpy arrays podem vir como (Batch, H, W, C) ou (Batch, C, H, W) dependendo do wrapper.
        # O wrapper GrayScale + Resize geralmente deixa (C, H, W). Se houver dimensão extra no array numpy de states:
        if len(states.shape) == 5: states = states.squeeze(1)

        actions = torch.tensor([t['action'] for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.buffer], device=self.device)
        vals = torch.tensor([t['val'] for t in self.buffer], device=self.device)
        rewards = [t['reward'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]

        # Calcular Vantagens (GAE)
        advantages = []
        gae = 0

        # Precisamos do valor do próximo estado para o último passo
        next_val = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_val = vals[t]

        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + vals # Retorno real = Vantagem + Valor estimado

        # Normalizar vantagens (estabilidade)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(states, device=self.device)

        # Treino em Epochs
        total_loss_mean = 0
        total_entropy_mean = 0
        n_batches = len(states) // self.batch_size

        for _ in range(self.n_epochs):
            # Baralhar índices
            indices = np.random.permutation(len(states))

            for i in range(n_batches):
                idx = indices[i * self.batch_size : (i + 1) * self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Forward pass
                dist, values = self.policy(batch_states)
                values = values.squeeze()
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratio para clipping (pi_new / pi_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Surrogate Loss (PPO Clip)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()

                # Total Loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss_mean += loss.item()
                total_entropy_mean += entropy.item()

        # Limpar buffer
        self.buffer = []
        self.curr_step += 1 # Conta atualizações

        # Salvar modelo periodicamente
        if self.curr_step % 10 == 0:
            self.save()

        # Retorna médias para log
        return total_loss_mean / (self.n_epochs * n_batches), total_entropy_mean / (self.n_epochs * n_batches)

    def save(self):
        save_path = self.save_dir / f"ppo_mario_{self.curr_step}.chkpt"
        torch.save({
            'model': self.policy.state_dict(),
            'steps': self.curr_step
        }, save_path)
        print(f"Modelo salvo em {save_path}")