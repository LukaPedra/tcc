import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --- ROBUST MONKEY PATCH START ---
import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.vector.utils import spaces
from copy import deepcopy

# 1. Patch for Box Spaces (Observations/Screen)
def _patched_batch_space_box(space, n=1):
    # Manually stack bounds for the batch
    low = np.stack([space.low] * n)
    high = np.stack([space.high] * n)
    return Box(low=low, high=high, dtype=space.dtype)

# 2. Patch for Discrete Spaces (Actions/Buttons)
def _patched_batch_space_discrete(space, n=1):
    # A batch of Discrete actions becomes a MultiDiscrete space
    return MultiDiscrete(np.full((n,), space.n, dtype=space.dtype))

# FORCE UPDATE the registry for both types
spaces.batch_space.register(Box, _patched_batch_space_box)
spaces.batch_space.register(Discrete, _patched_batch_space_discrete)
# --- ROBUST MONKEY PATCH END ---

import random, datetime
import argparse
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from gym.vector import AsyncVectorEnv

from logging_local import MetricLogger
from util import ResizeObservation, SkipFrame, StuckPenalty

# Import your agents here
from agent import Mario

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mario Agent")

    # Environment Settings
    parser.add_argument('--world', type=int, default=1, help='Super Mario World (1-8)')
    parser.add_argument('--stage', type=int, default=1, help='Super Mario Stage (1-4)')
    parser.add_argument('--action_mode', type=str, default='right_only', choices=['right_only', 'simple', 'complex'], help='Action space complexity')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments to run')

    # Training Settings
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'random', 'ppo'], help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=40000, help='Number of episodes to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--save_every', type=int, default=100000, help='Steps between model saves')

    # Hyperparameters
    parser.add_argument('--burnin', type=int, default=1e5, help='Burn-in steps before training')
    parser.add_argument('--learn_every', type=int, default=3, help='Steps between learning updates')
    parser.add_argument('--sync_every', type=int, default=1e4, help='Steps between target net syncs')
    parser.add_argument('--exploration_rate_decay', type=float, default=0.99995, help='Epsilon decay rate')

    return parser.parse_args()

def get_action_space(env_name, mode):
    if mode == 'right_only':
        return [['right'], ['right', 'A']]
    elif mode == 'simple':
        return gym_super_mario_bros.actions.SIMPLE_MOVEMENT
    elif mode == 'complex':
        return gym_super_mario_bros.actions.COMPLEX_MOVEMENT
    return [['right'], ['right', 'A']]

def make_env(env_name, actions):
    """Returns a callable that creates the environment."""
    def _init():
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, actions)

        # Apply Custom Wrapper Here
        # env = StuckPenalty(env, penalty=-1.0, threshold=60, terminal_on_stuck=False)

        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = TransformObservation(env, f=lambda x: x / 255.)
        env = FrameStack(env, num_stack=4)
        return env
    return _init

def main():
    args = parse_args()

    # 1. Initialize Vectorized Environment
    env_name = f'SuperMarioBros-{args.world}-{args.stage}-v0'
    actions = get_action_space(env_name, args.action_mode)

    # Create a list of environment constructors
    if args.num_envs > 1:
        env_fns = [make_env(env_name, actions) for _ in range(args.num_envs)]
        env = AsyncVectorEnv(env_fns)
        print(f"Initialized {args.num_envs} parallel environments.")
    else:
        # Fallback to single env if num_envs=1
        env = make_env(env_name, actions)()
        print("Initialized single environment.")

    state = env.reset()

    # 2. Setup Logging
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(save_dir)

    # 3. Initialize Agent
    if hasattr(env, 'single_action_space'):
        action_dim = env.single_action_space.n
    else:
        action_dim = env.action_space.n

    state_dim = (4, 84, 84)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    agent = None
    if args.agent == 'dqn':
        agent_args = vars(args).copy()
        if 'checkpoint' in agent_args: del agent_args['checkpoint']

        agent = Mario(
            state_dim=state_dim,
            action_dim=action_dim,
            save_dir=save_dir,
            checkpoint=checkpoint,
            **agent_args
        )
    elif args.agent == 'random':
        class RandomAgent:
            def __init__(self, action_dim): self.action_dim = action_dim
            def act(self, state):
                if isinstance(state, np.ndarray) and state.ndim == 4:
                    return np.random.randint(0, self.action_dim, size=state.shape[0])
                return random.randint(0, self.action_dim - 1)
            def cache(self, *args): pass
            def learn(self): return None, None
            @property
            def curr_step(self): return 0
            @property
            def exploration_rate(self): return 0
        agent = RandomAgent(action_dim)
    else:
        raise ValueError(f"Agent {args.agent} not implemented.")

    # 4. Training Loop
    print(f"Starting training with {args.agent} on {env_name}...")

    # Vectors for tracking parallel environments
    curr_rewards = np.zeros(args.num_envs)
    curr_lengths = np.zeros(args.num_envs)
    curr_losses = np.zeros(args.num_envs) # Accumulate loss
    curr_qs = np.zeros(args.num_envs)     # Accumulate Q-val

    try:
        while True:
            # Run agent on the batch of states
            action = agent.act(state)

            # Agent performs action in all envs simultaneously
            next_state, reward, done, info = env.step(action)

            # Handle Single vs Vector Env formatting differences
            if args.num_envs == 1:
                actual_next_state = np.array([next_state])
                infos = [info]
                dones = [done]
                rewards_batch = [reward]
            else:
                actual_next_state = next_state.copy()
                infos = info
                dones = done
                rewards_batch = reward

            # Update local trackers
            curr_rewards += rewards_batch
            curr_lengths += 1

            # Correct handling for Vector Env Auto-Reset
            for i, (d, inf) in enumerate(zip(dones, infos)):
                if d:
                    # If done, the next_state is the reset state.
                    # We need the terminal state for the cache.
                    if "terminal_observation" in inf:
                        actual_next_state[i] = inf["terminal_observation"]

                    # --- LOGGING FIX ---
                    # Push this completed episode to the logger
                    logger.ep_rewards.append(curr_rewards[i])
                    logger.ep_lengths.append(curr_lengths[i])

                    # Approximate episode loss/q by averaging the accumulated batch stats
                    avg_loss = curr_losses[i] / curr_lengths[i] if curr_lengths[i] > 0 else 0
                    avg_q = curr_qs[i] / curr_lengths[i] if curr_lengths[i] > 0 else 0
                    logger.ep_avg_losses.append(avg_loss)
                    logger.ep_avg_qs.append(avg_q)

                    # Reset trackers for this specific environment
                    curr_rewards[i] = 0
                    curr_lengths[i] = 0
                    curr_losses[i] = 0
                    curr_qs[i] = 0

            # Remember & Learn
            if args.num_envs == 1:
                agent.cache(state, next_state, action, reward, done)
            else:
                agent.cache(state, actual_next_state, action, reward, done)

            q, loss = agent.learn()

            # Accumulate loss/q for logging (apply batch avg to all active envs)
            if loss is not None:
                curr_losses += loss
            if q is not None:
                curr_qs += q

            state = next_state

            # Record Log to file/console periodically
            if agent.curr_step % 2000 == 0:
                 # Calculate approx episode number based on total finished episodes in logger
                 current_ep_count = len(logger.ep_rewards)
                 logger.record(
                    episode=current_ep_count,
                    epsilon=getattr(agent, 'exploration_rate', 0),
                    step=getattr(agent, 'curr_step', 0)
                )

            # Approx stop condition
            if agent.curr_step > args.episodes * 200:
                break

    except KeyboardInterrupt:
        print("Training interrupted. Saving...")
        agent.save()
        env.close()

if __name__ == '__main__':
    main()