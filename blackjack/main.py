from blackjack.Agents import BlackjackAgent, MonteCarloBlackjackAgent
from tqdm import tqdm
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np

# Training hyperparameters
learning_rate = 0.1
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# Create environments
env_q = gym.make("Blackjack-v1", sab=False)
env_q = gym.wrappers.RecordEpisodeStatistics(env_q, buffer_length=n_episodes)
env_mc = gym.make("Blackjack-v1", sab=False)
env_mc = gym.wrappers.RecordEpisodeStatistics(env_mc, buffer_length=n_episodes)

# Create agents
q_agent = BlackjackAgent(
    env=env_q,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
mc_agent = MonteCarloBlackjackAgent(
    env=env_mc,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Train Q-Learning agent
for episode in tqdm(range(n_episodes), desc="Q-Learning"):
    obs, info = env_q.reset()
    done = False
    while not done:
        action = q_agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env_q.step(action)
        q_agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    q_agent.decay_epsilon()

# Train Monte Carlo agent
for episode in tqdm(range(n_episodes), desc="Monte Carlo"):
    obs, info = env_mc.reset()
    done = False
    episode_history = []
    while not done:
        action = mc_agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env_mc.step(action)
        episode_history.append((obs, action, reward))
        done = terminated or truncated
        obs = next_obs
    mc_agent.update_episode(episode_history)
    mc_agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

# Q-Learning rewards
axs[0].set_title("Q-Learning: Episode rewards")
q_reward_moving_average = get_moving_avgs(env_q.return_queue, rolling_length, "valid")
axs[0].plot(range(len(q_reward_moving_average)), q_reward_moving_average, label="Q-Learning")
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")
axs[0].legend()

# Monte Carlo rewards
axs[1].set_title("Monte Carlo: Episode rewards")
mc_reward_moving_average = get_moving_avgs(env_mc.return_queue, rolling_length, "valid")
axs[1].plot(range(len(mc_reward_moving_average)), mc_reward_moving_average, label="Monte Carlo", color="orange")
axs[1].set_ylabel("Average Reward")
axs[1].set_xlabel("Episode")
axs[1].legend()

plt.tight_layout()
plt.show()

print(f"Q-Learning final average reward: {q_reward_moving_average[-1]}")
print(f"Monte Carlo final average reward: {mc_reward_moving_average[-1]}")