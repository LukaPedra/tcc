import gym
import numpy as np
from gym.spaces import Box
from skimage import transform

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
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class StuckPenalty(gym.Wrapper):
    def __init__(self, env, penalty=-2.0, threshold=60, terminal_on_stuck=True):
        """
        punishes the agent if x_pos doesn't change significantly
        over a window of 'threshold' frames.
        """
        super().__init__(env)
        self.penalty = penalty
        self.threshold = threshold
        self.terminal_on_stuck = terminal_on_stuck # <--- THIS IS YOUR LOW CAP
        self.x_history = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if 'x_pos' in info:
            self.x_history.append(info['x_pos'])

        if len(self.x_history) > self.threshold:
            self.x_history.pop(0)

            # If we haven't moved 2 pixels in the last 60 frames
            if max(self.x_history) - min(self.x_history) < 2:
                reward += self.penalty

                # KILL THE EPISODE IMMEDIATELY
                if self.terminal_on_stuck:
                    done = True

        return state, reward, done, info

    def reset(self, **kwargs):
        self.x_history = []
        return self.env.reset(**kwargs)