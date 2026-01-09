import torch
import math


class StateScaler:
    """
    Converts raw environment states (dict) into a normalized, model-ready tensor.
    - Cyclic encodings for time features (minute-of-day, day-of-week, month).
    - Z-score normalization for energy generation/consumption and prices.
    - Battery charge scaled to [0, 1] using known capacity.
    """
    def __init__(self, env, device, eps: float = 1e-6):
        self.device = device
        self.eps = eps

        energy_df = env.consumption_generation_data
        prices_df = env.prices_data

        self.energy_mean = energy_df[["energy_generation", "energy_consumption"]].mean()
        self.energy_std = (energy_df[["energy_generation", "energy_consumption"]].std() + eps)

        self.price_mean = prices_df[["energy_sell_price", "energy_buy_price"]].mean()
        self.price_std = (prices_df[["energy_sell_price", "energy_buy_price"]].std() + eps)

        battery_space = env.observation_space["battery_charge"]
        self.battery_capacity = float(battery_space.high[0])

        # 5 numeric features + 3 cyclic pairs (6 values) = 11 total inputs
        self.feature_dim = 11

    def _encode_cyclic(self, value: float, period: float):
        angle = 2 * math.pi * (value % period) / period
        return math.sin(angle), math.cos(angle)

    def _zscore(self, value, mean, std):
        return (float(value) - float(mean)) / float(std)

    def transform(self, state: dict) -> torch.Tensor:
        minute_sin, minute_cos = self._encode_cyclic(state["time"], 24 * 60)
        weekday_sin, weekday_cos = self._encode_cyclic(state["day_of_week"], 7)
        month_sin, month_cos = self._encode_cyclic(state["month"], 12)
        energy_generation = self._zscore(state["energy_generation"], self.energy_mean["energy_generation"], self.energy_std["energy_generation"])
        energy_consumption = self._zscore(state["energy_consumption"], self.energy_mean["energy_consumption"], self.energy_std["energy_consumption"])
        energy_sell_price = self._zscore(state["energy_sell_price"], self.price_mean["energy_sell_price"], self.price_std["energy_sell_price"])
        energy_buy_price = self._zscore(state["energy_buy_price"], self.price_mean["energy_buy_price"], self.price_std["energy_buy_price"])
        battery_charge = float(state["battery_charge"]) / self.battery_capacity

        features = [
            energy_generation,
            energy_consumption,
            battery_charge,
            energy_sell_price,
            energy_buy_price,
            minute_sin,
            minute_cos,
            weekday_sin,
            weekday_cos,
            month_sin,
            month_cos,
        ]
        return torch.tensor(features, dtype=torch.float, device=self.device)