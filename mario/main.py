import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from Agents import MarioDQNAgent
from utils import create_mario_env
from tqdm import tqdm
import torch

# 1. Create and wrap the environment
env = create_mario_env()
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 2. Initialize Agent
agent = MarioDQNAgent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# 3. Training Loop
num_episodes = 1000
target_update_frequency = 1000 # frames

frame_count = 0
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        env.render()

        agent.remember(state, action, reward, next_state, done)

        agent.replay()

        agent.decay_epsilon()

        state = next_state
        total_reward += reward
        frame_count += 1

        if frame_count % target_update_frequency == 0:
            agent.update_target_net()

    print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

env.close()
