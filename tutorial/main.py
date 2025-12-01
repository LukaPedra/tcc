import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import argparse
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

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

    # Training Settings
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'random', 'ppo'], help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=40000, help='Number of episodes to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')

    # Hyperparameters (Passed to Agent)
    parser.add_argument('--burnin', type=int, default=100000, help='Burn-in steps before training')
    parser.add_argument('--learn_every', type=int, default=3, help='Steps between learning updates')
    parser.add_argument('--sync_every', type=int, default=10000, help='Steps between target net syncs')

    return parser.parse_args()

def get_action_space(env_name, mode):
    # Helper to determine action space based on complexity mode
    if mode == 'right_only':
        return [['right'], ['right', 'A']]
    elif mode == 'simple':
        return gym_super_mario_bros.actions.SIMPLE_MOVEMENT
    elif mode == 'complex':
        return gym_super_mario_bros.actions.COMPLEX_MOVEMENT
    return [['right'], ['right', 'A']]

def main():
    args = parse_args()

    # 1. Initialize Environment
    env_name = f'SuperMarioBros-{args.world}-{args.stage}-v0'
    env = gym_super_mario_bros.make(env_name)

    actions = get_action_space(env_name, args.action_mode)
    env = JoypadSpace(env, actions)
    env = StuckPenalty(env, penalty=-5.0, threshold=60)

    # Apply Wrappers
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    # 2. Setup Logging
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(save_dir)

    # 3. Initialize Agent
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    agent = None
    if args.agent == 'dqn':
        # Create a copy of args to modify safely
        agent_args = vars(args).copy()

        # FIX: Remove 'checkpoint' from kwargs because we pass it explicitly
        if 'checkpoint' in agent_args:
            del agent_args['checkpoint']

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
            def act(self, state): return random.randint(0, self.action_dim - 1)
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
    print(f"Starting training with {args.agent} on {env_name}")

    for e in range(args.episodes):
        state = env.reset()

        while True:
            # Run agent on the state
            action = agent.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember & Learn
            agent.cache(state, next_state, action, reward, done)
            q, loss = agent.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=getattr(agent, 'exploration_rate', 0),
                step=getattr(agent, 'curr_step', 0)
            )

if __name__ == '__main__':
    main()