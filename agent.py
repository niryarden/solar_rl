import torch
from torch import nn
from tqdm import tqdm
import wandb

import json
import random
from datetime import datetime
import itertools
import os

from experience_replay import ReplayMemory
from dqn import DQN
from environment import EnergyHouseholdEnv
from state_scaler import StateScaler


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "run_outputs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent():
    def __init__(self, hyperparameter_set, log_wandb=False):
        with open('hyperparameters.json', 'r') as file:
            all_hyperparameter_sets = json.load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of each training batch
        self.warmup_steps       = hyperparameters['warmup_steps']           # number of steps to fill the replay memory
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.policy_update_rate = hyperparameters['policy_update_rate']     # number of steps to update the policy network
        self.target_sync_rate   = hyperparameters['target_sync_rate']       # number of steps to sync the target network
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.learning_rate      = hyperparameters['learning_rate']          # learning rate
        self.discount_factor    = hyperparameters['discount_factor']        # discount rate (gamma)
        self.hidden_dim         = hyperparameters['hidden_dim']             # DQN width

        self.wandb_run = None
        if log_wandb:
            self.wandb_run = wandb.init(
                entity="nir-yarden-hebrew-university-of-jerusalem",
                project="solar-rl",
                name=self.hyperparameter_set,
                config=hyperparameters,
            )

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None  # Initialize later.

        run_dir = os.path.join(RUNS_DIR, self.hyperparameter_set)
        os.makedirs(run_dir, exist_ok=True)
        self.MODEL_FILE = os.path.join(run_dir, f'{self.hyperparameter_set}.pt')

    def train(self):
        env = EnergyHouseholdEnv()
        state_scaler = StateScaler(env, device)

        nn_input_dim = state_scaler.feature_dim
        nn_output_dim = env.action_space.n
        
        policy_dqn = DQN(nn_input_dim, nn_output_dim, self.hidden_dim).to(device)
        target_dqn = DQN(nn_input_dim, nn_output_dim, self.hidden_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        epsilon = self.epsilon_init
        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
        
        step_count = 0  # counts environment steps across episodes
        best_reward = -9999999
        last_saved_episode = -1
        last_30_acc_rewards = []

        # Train indefinitely, until manually stop
        for episode in itertools.count():
            state_dict, _ = env.reset()
            state = state_scaler.transform(state_dict)
            terminated = False
            episode_acc_reward = 0.0

            while not terminated:
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select policy action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state_dict, reward, terminated, _, _ = env.step(action.item())
                episode_acc_reward += reward

                new_state = state_scaler.transform(new_state_dict)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                memory.append((state, action, new_state, reward, terminated))
                step_count += 1
                state = new_state
                
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                if step_count >= self.warmup_steps:
                    if len(memory) >= self.mini_batch_size and step_count % self.policy_update_rate == 0:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                    
                    if step_count % self.target_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

            if episode_acc_reward > best_reward or episode - last_saved_episode >= 50:
                print(f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_acc_reward:.2f} at episode {episode}, saving model...")                
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                last_saved_episode = episode
                if episode_acc_reward > best_reward:
                    best_reward = episode_acc_reward
            
            last_30_acc_rewards.append(episode_acc_reward)
            last_30_acc_rewards = last_30_acc_rewards[-30:]
            
            if self.wandb_run:
                self.wandb_run.log({
                    "episode": episode,
                    "reward": episode_acc_reward,
                    "average_last_30_episode_rewards": sum(last_30_acc_rewards) / len(last_30_acc_rewards)
                })
    

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Double DQN target: action selection by policy network, evaluation by target network
        with torch.no_grad():
            next_actions = policy_dqn(new_states).argmax(dim=1, keepdim=True)
            next_target_q = target_dqn(new_states).gather(dim=1, index=next_actions).squeeze()
            target_q = rewards + (1 - terminations) * self.discount_factor * next_target_q

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()       # Update network parameters


    def evaluate(self):
        env = EnergyHouseholdEnv()
        state_scaler = StateScaler(env, device)

        nn_input_dim = state_scaler.feature_dim
        nn_output_dim = env.action_space.n

        policy_dqn = DQN(nn_input_dim, nn_output_dim, self.hidden_dim).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()

        acc_reward = 0.0
        acc_reward_memory = []
        state_dict, _ = env.reset(is_train=False, seed=42)
        state = state_scaler.transform(state_dict)
        terminated = False
        step_count = 0
        cur_month = state_dict["month"]

        with tqdm(total=12, initial=1) as pbar:
            while not terminated:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state_dict, reward, terminated, _, _ = env.step(action.item())
                if new_state_dict['month'] > cur_month:
                    pbar.update(1)
                cur_month = new_state_dict["month"]
                state = state_scaler.transform(new_state_dict)
                acc_reward += reward
                step_count += 1
                if step_count % 1440 == 0:
                    acc_reward_memory.append(acc_reward)
        
        return acc_reward_memory
