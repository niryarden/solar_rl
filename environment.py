import gymnasium as gym
from gymnasium import spaces
import numpy as np

# https://www.nature.com/articles/s41597-025-04747-w
CONSUMPTION_GENERATION_DATASET_PATH = "datasets/consumption_generation.csv"
# https://data.nordpoolgroup.com/auction/day-ahead/prices?deliveryAreas=EE
PRICES_DATASET_PATH = "datasets/prices.csv"

"""
State space:
- Energy generation (kwh)
- Energy consumption (kwh)
- Battery charge (kwh)
- Energy sell price (€/kwh)
- Energy buy price (€/kwh)
- Time
- Day of week
- Month
"""

"""
Action space:
- Charge battery (kwh)
- Discharge battery (kwh)
- Do nothing
"""

class EnergyHouseholdEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(3, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.random.rand(3).astype(np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info