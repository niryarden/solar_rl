# Energy Household RL Agent

This project implements a reinforcement learning (RL) agent to optimize energy management in a simulated household environment. The agent aims to make decisions that maximize accumulated reward by efficiently managing energy consumption, generation, storage, and trading.

## Run the Project
### 1. Install requirements

```bash
pip install -r requirements.txt
```

### Evaluating our pre-trained agent

To evaluate our pre-trained agent in comparison to the baselines:

```bash
python main.py --hyperparameters results_run
```
The output of this run is the plot that can be found in `run_outputs/results_run/accumulated_reward.png`

### 2. Training your own model

Optional: Edit the hyperparameters of 'example_run' in `hyperparameters.json`:
```json
    "example_run": {
        "mini_batch_size": 8,
        "warmup_steps": 1000,
        "replay_memory_size": 10000,
        "policy_update_rate": 1,
        "target_sync_rate": 128,
        "epsilon_init": 1,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.05,
        "learning_rate": 0.0001,
        "discount_factor": 0.99,
        "hidden_dim": 32
    }
```

Train the RL agent using the defined hyperparameters:

```bash
python main.py --train --hyperparameters example_run
```

This training process will continue endlessly until the you kill it.
Its output will be the neural network's weights in `run_outputs/example_run/example_run.pt`

You may now evalute your model using 
```bash
python main.py --hyperparameters example_run
```

## Code

- `main.py`: Entry point for training and evaluation.
- `agent.py`: Implementation of the RL agent.
- `environment.py`: Simulated household energy environment (state and dynamics).
- `state_scaler.py`: State normalization and encoding.
- `baselines.py`: Baseline policy implementations.
- `hyperparameters.json`: Predefined sets of hyperparameters.

