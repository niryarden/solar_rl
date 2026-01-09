import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import json
import random
from datetime import datetime, timedelta
import argparse
import itertools
import os

from experience_replay import ReplayMemory
from dqn import DQN
from environment import EnergyHouseholdEnv


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "run_outputs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.json', 'r') as file:
            all_hyperparameter_sets = json.load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.learning_rate      = hyperparameters['learning_rate']          # learning rate
        self.discount_factor    = hyperparameters['discount_factor']        # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target networks
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of each training batch
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.hidden_dim         = hyperparameters['hidden_dim']             # DQN width

        self.loss_fn = nn.MSELoss()          # Mean Squared Error Loss function
        self.optimizer = None                # NN Optimizer. Initialize later.

        run_dir = os.path.join(RUNS_DIR, self.hyperparameter_set)
        os.makedirs(run_dir, exist_ok=True)
        self.LOG_FILE   = os.path.join(run_dir, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(run_dir, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(run_dir, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = EnergyHouseholdEnv()
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.hidden_dim).to(device)

        if is_training:
            # init for training
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.hidden_dim).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            # Init for Evaluation
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Train indefinitely, until manually stop
        for episode in itertools.count():
            state, _ = env.reset() 
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_acc_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = env.step(action.item())
                episode_acc_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_acc_reward)
            # Save model when new best reward is obtained.
            if is_training:
                if episode_acc_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_acc_reward:0.1f} ({(episode_acc_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_acc_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # Only when memory is large enough to populate a batch
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Calculate target Q values (gt) and cur_policy's Q values.
        # compare them to calculate the loss
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters
