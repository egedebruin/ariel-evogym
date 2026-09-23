import numpy as np
import gymnasium as gym
from evogym.envs import EvoGymBase
from evogym import EvoWorld, utils

class WalkerLongEnv(EvoGymBase):
    def __init__(self, world, render_mode=None):
        super().__init__(world, render_mode=render_mode)

    def reset(self, **kwargs):
        super().reset()
        return None, {}

    def step(self, action):
        super().step({"robot": action})
        return None, 0.0, False, False, {}
