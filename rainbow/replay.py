import torch
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario
import argparse
from pathlib import Path
import time

# Function to setup the environment exactly like main.py
def make_env():
    # 1. Initialize Environment
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

    # Apply wrappers
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)

    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    return env

def run_replay(checkpoint_path):
    env = make_env()

    # Initialize Agent
    # save_dir is not used for logging in replay, but required by init
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=Path("."))

    # Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load state dictionary into the model
    mario.net.load_state_dict(checkpoint['model'])

    # CRITICAL: Set model to evaluation mode.
    # This ensures NoisyLinear layers use the mean weights (exploitation)
    # instead of adding noise (exploration).
    mario.net.eval()

    # Bypass the burnin period in agent.act so it uses the network immediately
    mario.curr_step = mario.burnin + 10

    # Replay Loop
    state = env.reset()
    total_reward = 0

    print("Replay started...")
    while True:
        # We don't need mario.cache or mario.learn here, just act
        action = mario.act(state)

        # Standard Gym Step
        next_state, reward, done, trunc, info = env.step(action)

        total_reward += reward
        state = next_state

        # Render the game
        env.render()

        # Slow down slightly for human viewing if needed
        time.sleep(0.02)

        if done or info["flag_get"]:
            break

    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a trained Rainbow Mario Agent")
    parser.add_argument("checkpoint", type=str, help="Path to the .chkpt file")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
    else:
        run_replay(args.checkpoint)