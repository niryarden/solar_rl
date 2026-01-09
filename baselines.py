from environment import EnergyHouseholdEnv
from enum import Enum
from tqdm import tqdm

class BaselineName(Enum):
    RANDOM = "random"
    BY_TIME = "by_time"
    BY_PRICE = "by_price"
    BY_USAGE = "by_usage"

class Baseline:
    def __init__(self, baseline_name: BaselineName):
        self.baseline_name = baseline_name
        self.env = EnergyHouseholdEnv()
    
    def run(self):
        acc_reward = 0.0
        acc_reward_memory = []
        state, _ = self.env.reset(is_train=False, seed=42)
        terminated = False
        step_count = 0
        with tqdm(total=12, initial=1) as pbar:
            while not terminated:
                action = self._get_action(state)
                new_state, reward, terminated, _, _ = self.env.step(action)
                if new_state['month'] > state['month']:
                    pbar.update(1)
                state = new_state
                acc_reward += reward
                step_count += 1
                if step_count % 1440 == 0:
                    acc_reward_memory.append(acc_reward)
        
        return acc_reward_memory
    
    def _get_action(self, state):
        if self.baseline_name == BaselineName.RANDOM:
            return self._get_action_random(state)
        elif self.baseline_name == BaselineName.BY_TIME:
            return self._get_action_by_time(state)
        elif self.baseline_name == BaselineName.BY_PRICE:
            return self._get_action_by_price(state)
        elif self.baseline_name == BaselineName.BY_USAGE:
            return self._get_action_by_usage(state)
        elif self.baseline_name == BaselineName.OFFGRID:
            return self._get_action_offgrid(state)
    
    def _get_action_random(self, state):
        return self.env.action_space.sample()
    
    def _get_action_by_time(self, state):
        """
        use_first: (7:00 to 9:00) AND (18:00 to 23:00)
        sell_first: (9:00 to 18:00) AND (23:00 to 7:00)
        """
        time_in_minutes = state["time"]
        time_in_hours = time_in_minutes // 60
        if time_in_hours >= 7 and time_in_hours <= 9:
            return 0
        elif time_in_hours >= 18 and time_in_hours <= 23:
            return 0
        else:
            return 1
    
    def _get_action_by_price(self, state):
        """
        sell_first: sell_price > 0.1 euro/kWh
        use_first: sell_price < 0.1 euro/kWh
        """
        sell_price = state["energy_sell_price"]
        if sell_price > 0.1:
            return 1
        else:
            return 0
    
    def _get_action_by_usage(self, state):
        """
        sell_first: consumption < 1.5 kWh
        use_first: consumption > 1.5 kWh
        """
        consumption = state["energy_consumption"]
        if consumption < 1.5:
            return 1
        else:
            return 0
