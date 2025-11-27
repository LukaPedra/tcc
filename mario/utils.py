import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym_super_mario_bros

def create_mario_env(world=1, stage=1):
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
    # Apply wrappers
    # 1. Convert to grayscale, but don't keep the channel dimension
    env = GrayScaleObservation(env, keep_dim=False)
    # 2. Resize to 84x84
    env = ResizeObservation(env, shape=84)
    # 3. Stack 4 frames together
    env = FrameStack(env, num_stack=4)
    return env
