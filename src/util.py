import gym
import numpy as np
import torch
from gym.spaces import Box
from torchvision import transforms as T

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        # Keep as uint8 to save memory and avoid corruption in VectorEnv
        observation = torch.tensor(observation.copy(), dtype=torch.uint8)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # REMOVED Normalize. We do it in the agent now.
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class StuckPenalty(gym.Wrapper):
    def __init__(self, env, penalty=-2.0, threshold=60, terminal_on_stuck=False):
        super().__init__(env)
        self.penalty = penalty
        self.threshold = threshold
        self.terminal_on_stuck = terminal_on_stuck
        self.x_history = []

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        # Handle 'info' being a list in VectorEnvs or dict in single envs
        # Since this wrapper wraps a Single env (before vectorization), info is a dict.
        if 'x_pos' in info:
            self.x_history.append(info['x_pos'])

        if len(self.x_history) > self.threshold:
            self.x_history.pop(0)
            if max(self.x_history) - min(self.x_history) < 2:
                reward += self.penalty
                if self.terminal_on_stuck:
                    done = True

        return state, reward, done, trunc, info

    def reset(self, **kwargs):
        self.x_history = []
        return self.env.reset(**kwargs)