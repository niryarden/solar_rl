import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# https://www.nature.com/articles/s41597-025-04747-w
CONSUMPTION_GENERATION_DATASET_PATH = "datasets/consumption_generation.csv"
# https://data.nordpoolgroup.com/auction/day-ahead/prices?deliveryAreas=EE
PRICES_DATASET_PATH = "datasets/prices.csv"

# Days in each month (ignoring leap years for simplicity)
DAYS_IN_MONTH = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}
YEAR = 2023

"""
State space:
- Energy generation (kwh)
- Energy consumption (kwh)
- Battery charge (kwh)
- Energy sell price (€/kwh)
- Energy buy price (€/kwh)
- Time (minutes since midnight)
- Day of week (0-6)
- Month (0-11)
"""

"""
Action space:
- Use
- Sell
- Store
"""

"""
Reward:
- Possive if profit is made from selling energy
- Negative if loss is incurred from buying energy
"""

BATTERY_CAPACITY = 15.0 # kwh
INIT_BATTERY_CHARGE = 5.0 # kwh


class EnergyHouseholdEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "energy_generation": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "energy_consumption": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "battery_charge": spaces.Box(low=0, high=BATTERY_CAPACITY, shape=(1,), dtype=np.float32),
            "energy_sell_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "energy_buy_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "time": spaces.Discrete(24 * 60), # minutes since midnight
            "day_of_week": spaces.Discrete(7),
            "month": spaces.Discrete(12),
        })
        self.action_space = spaces.Discrete(3)
        self._load_datasets()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_time()
        self._init_battery()
        self._update_prices()
        self._update_energy()
        return self._get_obs(), self._get_info()

    def step(self, action):
        terminated = self._advance_time()
        self._update_prices()
        self._update_energy()
        self._update_battery(action)
        obs = self._get_obs()
        info = self._get_info()
        reward = 0.0
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        timestamp = self.cur_timestamp
        obs = {
            "time": timestamp.hour * 60 + timestamp.minute,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month - 1,
        }
        obs |= self.cur_prices | self.cur_energy | self.cur_battery
        return obs
    
    def _get_info(self):
        return {}
    
    def _load_datasets(self):
        """
        This code loads household consumption and generation data, and energy price data from CSVs. It:
        - parses and normalizes timestamps (removing timezones)
        - averages rows per minute (groupby + mean)
        - resamples to fill any missing minute-intervals (ffill)
        - aligns and fills the price data index to match the energy data's timestamps
        This ensures every energy observation has a corresponding, synchronized price/energy value.
        """
        self.consumption_generation_data = (
            pd.read_csv(CONSUMPTION_GENERATION_DATASET_PATH, header=0, parse_dates=[0])
            .set_axis(["timestamp", "energy_consumption", "energy_generation"], axis=1)
        )
        self.consumption_generation_data["timestamp"] = pd.to_datetime(
            self.consumption_generation_data["timestamp"]
        ).dt.floor("min")
        self.consumption_generation_data = (
            self.consumption_generation_data
            .groupby("timestamp", as_index=True)[["energy_consumption", "energy_generation"]]
            .mean()
            .sort_index()
        )

        self.prices_data = (
            pd.read_csv(PRICES_DATASET_PATH, header=0, parse_dates=[0])
            .set_axis(["timestamp", "energy_sell_price", "energy_buy_price"], axis=1)
        )
        self.prices_data["timestamp"] = pd.to_datetime(
            self.prices_data["timestamp"], utc=True
        ).dt.tz_convert(None)
        self.prices_data = (
            self.prices_data
            .set_index("timestamp")
            .groupby(level=0, sort=False)[["energy_sell_price", "energy_buy_price"]]
            .mean()
            .sort_index()
            .resample("min")
            .ffill()
        )
        energy_index = self.consumption_generation_data.index
        self.prices_data = (
            self.prices_data
            .reindex(energy_index, method="ffill")
            .bfill()
        )
    
    def _init_time(self):
        self.cur_timestamp = self.consumption_generation_data.index[0]
    
    def _advance_time(self):
        self.cur_timestamp += pd.Timedelta(minutes=1)
        return self.cur_timestamp.year > YEAR
    
    def _init_battery(self):
        self.cur_battery = {"battery_charge": INIT_BATTERY_CHARGE}
    
    def _update_battery(self, action):
        pass

    def _update_prices(self):
        row = self.prices_data.loc[self.cur_timestamp]
        energy_sell_price = float(row["energy_sell_price"])
        energy_buy_price = float(row["energy_buy_price"])
        self.cur_prices = {
            "energy_sell_price": energy_sell_price,
            "energy_buy_price": energy_buy_price,
        }

    def _update_energy(self):
        row = self.consumption_generation_data.loc[self.cur_timestamp]
        energy_generation = float(row["energy_generation"])
        energy_consumption = float(row["energy_consumption"])
        self.cur_energy = {
            "energy_generation": energy_generation,
            "energy_consumption": energy_consumption,
        }

    
def main():
    env = EnergyHouseholdEnv()
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 10:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_count}:")
        print(f"  Action taken: {action}")
        print(f"  Reward: {reward}")
        print("  Next observation:")
        for key, value in next_obs.items():
            print_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"    {key}: {print_value}")
        
        done = terminated or truncated
        step_count += 1
    if done:
        print("Finished with termination.")
    else:
        print("Hit step limit.")

if __name__ == "__main__":
    main()
