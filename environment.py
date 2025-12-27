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


BATTERY_CAPACITY = 15.0 # kwh
INIT_BATTERY_CHARGE = 5.0 # kwh


class EnergyHouseholdEnv(gym.Env):
    def __init__(self):
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
        """
            Action space:
            - Use
            - Sell
            - Store
        """
        self.action_space = spaces.Discrete(3)
        self._load_datasets()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_time()
        self._init_battery()
        self._align_prices_with_time()
        self._align_energy_with_time()
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = self._apply_action(action)
        terminated = self._advance_time()
        if not terminated:
            self._align_prices_with_time()
            self._align_energy_with_time()
        obs = self._get_obs()
        info = self._get_info()
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        timestamp = self.cur_timestamp
        obs = {}
        obs |= {
            "time": timestamp.hour * 60 + timestamp.minute,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month - 1,
        }
        obs |= { "battery_charge": self.cur_battery }
        obs |= self.cur_prices
        obs |= self.cur_energy
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
        self.cur_battery = INIT_BATTERY_CHARGE

    def _align_prices_with_time(self):
        row = self.prices_data.loc[self.cur_timestamp]
        energy_sell_price = float(row["energy_sell_price"])
        energy_buy_price = float(row["energy_buy_price"])
        self.cur_prices = {
            "energy_sell_price": energy_sell_price,
            "energy_buy_price": energy_buy_price,
        }

    def _align_energy_with_time(self):
        row = self.consumption_generation_data.loc[self.cur_timestamp]
        energy_generation = float(row["energy_generation"])
        energy_consumption = float(row["energy_consumption"])
        self.cur_energy = {
            "energy_generation": energy_generation,
            "energy_consumption": energy_consumption,
        }
    
    def _apply_action(self, action):
        """
            System Logic:
            - Energy consumption is always met.
            - The priority is: from generation (if allocated), then from battery, then from buying.
            
            Reward:
            - Possive if profit is made from selling energy
            - Negative if loss is incurred from buying energy
        """
        if action == 0:
            return self._apply_action_use()
        elif action == 1:
            return self._apply_action_sell()
        elif action == 2:
            return self._apply_action_store()
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _apply_action_use(self):
        consumption_rate_kwh = self.cur_energy["energy_consumption"]
        generation_rate_kwh = self.cur_energy["energy_generation"]
        net_deficit_rate_kwh = consumption_rate_kwh - generation_rate_kwh

        # 1. If generation covers consumption
        if net_deficit_rate_kwh <= 0:
            return 0  # no cost, no battery use

        energy_needed_this_minute = net_deficit_rate_kwh / 60
        battery_charge_kwh = self.cur_battery
        # 2. If battery can fully cover the deficit
        if battery_charge_kwh >= energy_needed_this_minute:
            self.cur_battery -= energy_needed_this_minute
            return 0

        # 3. If battery partially covers the deficit (buy power)
        energy_from_battery = battery_charge_kwh
        energy_to_buy = energy_needed_this_minute - energy_from_battery
        self.cur_battery = 0
        energy_buy_price = self.cur_prices["energy_buy_price"]
        cost = energy_to_buy * energy_buy_price
        return -cost
    
    def _apply_action_sell(self):
        consumption_rate_kwh = self.cur_energy["energy_consumption"]
        generation_rate_kwh = self.cur_energy["energy_generation"]
        sell_price = self.cur_prices["energy_sell_price"]
        buy_price = self.cur_prices["energy_buy_price"]

        energy_generated = generation_rate_kwh / 60
        energy_needed = consumption_rate_kwh / 60

        # 1. Sell all generated energy
        money_from_selling = energy_generated * sell_price

        # 2. Use battery to cover consumption
        battery_charge_kwh = self.cur_battery
        energy_from_battery = min(battery_charge_kwh, energy_needed)
        self.cur_battery -= energy_from_battery
        remaining_energy_needed = energy_needed - energy_from_battery

        # 3. Buy remaining energy from the grid
        energy_to_buy = remaining_energy_needed
        money_for_buying = energy_to_buy * buy_price
        return money_from_selling - money_for_buying
    
    def _apply_action_store(self):
        consumption_rate_kwh = self.cur_energy["energy_consumption"]
        generation_rate_kwh = self.cur_energy["energy_generation"]
        buy_price = self.cur_prices["energy_buy_price"]

        energy_generated = generation_rate_kwh / 60
        energy_consumed = consumption_rate_kwh / 60

        # 1. Charge battery with all generated energy
        self.cur_battery = min(self.cur_battery + energy_generated, BATTERY_CAPACITY)

        # 2. Buy all consumed energy from the grid
        energy_to_buy = energy_consumed
        money_for_buying = energy_to_buy * buy_price
        return -money_for_buying
        
    
def main():
    env = EnergyHouseholdEnv()
    acc_reward = 0.0
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 1000:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        acc_reward += reward
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
    print(f"Accumulated reward: {acc_reward}")

if __name__ == "__main__":
    main()
