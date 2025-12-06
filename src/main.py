import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --- MONKEY PATCH FOR GYM/NUMPY COMPATIBILITY ---
import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.vector.utils import spaces
from copy import deepcopy

def _patched_batch_space_box(space, n=1):
    low = np.stack([space.low] * n)
    high = np.stack([space.high] * n)
    return Box(low=low, high=high, dtype=space.dtype)

def _patched_batch_space_discrete(space, n=1):
    return MultiDiscrete(np.full((n,), space.n, dtype=space.dtype))

spaces.batch_space.register(Box, _patched_batch_space_box)
spaces.batch_space.register(Discrete, _patched_batch_space_discrete)
# --- END PATCH ---

import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from gym.vector import AsyncVectorEnv
import datetime
from pathlib import Path

# Local imports
from util import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario
from logging_local import MetricLogger

def make_env(env_name, actions):
    """Factory function for creating environments"""
    def _init():
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(env_name, new_step_api=True)
        else:
            env = gym_super_mario_bros.make(env_name, render_mode='rgb', apply_api_compatibility=True)

        env = JoypadSpace(env, actions)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        if gym.__version__ < '0.26':
            env = FrameStack(env, num_stack=4, new_step_api=True)
        else:
            env = FrameStack(env, num_stack=4)
        return env
    return _init

def main():
    # --- CONFIG ---
    NUM_ENVS = 4
    ENV_NAME = "SuperMarioBros-1-1-v0"
    ACTIONS = [["right"], ["right", "A"]]

    # Initialize Vector Envs
    env_fns = [make_env(ENV_NAME, ACTIONS) for _ in range(NUM_ENVS)]
    env = AsyncVectorEnv(env_fns)

    # Setup
    state = env.reset()
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    if not use_cuda and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    # For Vector Env, action space is in single_action_space
    action_dim = env.single_action_space.n
    mario = Mario(state_dim=(4, 84, 84), action_dim=action_dim, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    episodes = 40000
    print(f"Starting training with {NUM_ENVS} parallel environments...")

    # Trackers for parallel envs
    curr_rewards = np.zeros(NUM_ENVS)
    curr_lengths = np.zeros(NUM_ENVS)
    curr_losses = np.zeros(NUM_ENVS)
    curr_qs = np.zeros(NUM_ENVS)

    try:
        while True:
            # 1. ACT
            action = mario.act(state)

            # 2. STEP (All envs at once)
            # FIX: Handle both 4-value (Old Gym) and 5-value (New Gym) returns
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, trunc, info = step_result
            else:
                next_state, reward, done, info = step_result

            # 3. HANDLE RESET & CACHE
            actual_next_state = next_state.copy()

            for i in range(NUM_ENVS):
                # Check if this specific env finished
                if done[i]:
                    # Extract terminal observation carefully based on 'info' structure
                    if isinstance(info, dict):
                        # Case A: info is a Dict of Lists (e.g. {'coins': [0,0,0,0]})
                        if "terminal_observation" in info:
                            actual_next_state[i] = info["terminal_observation"][i]
                        elif "final_observation" in info:
                            actual_next_state[i] = info["final_observation"][i]
                    else:
                        # Case B: info is a List/Tuple of Dicts (Standard)
                        if "terminal_observation" in info[i]:
                            actual_next_state[i] = info[i]["terminal_observation"]
                        elif "final_observation" in info[i]:
                             actual_next_state[i] = info[i]["final_observation"]

                    # LOGGING
                    avg_loss = curr_losses[i] / curr_lengths[i] if curr_lengths[i] > 0 else 0
                    avg_q = curr_qs[i] / curr_lengths[i] if curr_lengths[i] > 0 else 0

                    final_reward = curr_rewards[i] + reward[i]
                    final_length = curr_lengths[i] + 1

                    logger.log_episode_manual(final_reward, final_length, avg_loss, avg_q)

                    # Reset trackers
                    curr_rewards[i] = 0
                    curr_lengths[i] = 0
                    curr_losses[i] = 0
                    curr_qs[i] = 0
                else:
                    # Accumulate if not done
                    curr_rewards[i] += reward[i]
                    curr_lengths[i] += 1

            # 4. CACHE
            mario.cache(state, actual_next_state, action, reward, done)

            # 5. LEARN
            q, loss = mario.learn()

            # Accumulate metrics for active envs
            if loss is not None:
                curr_losses += loss
            if q is not None:
                curr_qs += q

            # 6. UPDATE STATE
            state = next_state

            # 7. LOGGING to console
            if mario.curr_step % 2000 == 0:
                # Estimate episode count
                current_ep_count = len(logger.ep_rewards)
                logger.record(episode=current_ep_count, epsilon=mario.exploration_rate, step=mario.curr_step)

            if mario.curr_step > episodes * 200: # Approx stop
                break

    except KeyboardInterrupt:
        print("Training interrupted. Saving...")
        mario.save()
        env.close()

if __name__ == '__main__':
    main()