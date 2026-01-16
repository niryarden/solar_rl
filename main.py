import argparse
import os
from agent import Agent
from baselines import Baseline, BaselineName
import matplotlib.pyplot as plt


RUNS_DIR = "run_outputs"
SHOW_PLOTS = os.getenv("COLAB_SHOW_PLOTS", "0") == "1"


def parse_args():
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--hyperparameters', type=str, required=True, help='Name of hyperparameters set from the hyperparameters.json file')
    parser.add_argument('--train', action='store_true', help='Training mode')
    return parser.parse_args()


def train(args):
    agent = Agent(hyperparameter_set=args.hyperparameters, log_wandb=args.wandb)
    agent.train()


def evaluate(args):
    acc_reward_memory_dict = {}
    for baseline_name in BaselineName:
        print(f"Evaluating baseline: {baseline_name.value}")
        baseline = Baseline(baseline_name=baseline_name)
        acc_reward_memory_dict[baseline_name.value] = baseline.evaluate()
    
    print("Evaluating agent")
    agent = Agent(hyperparameter_set=args.hyperparameters)
    acc_reward_memory_dict["agent"] = agent.evaluate()
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for key, value in acc_reward_memory_dict.items():
        ax.plot(
            value,
            label=key,
            linewidth=2.2,
            alpha=0.95,
        )

    ax.axhline(0, color="0.6", linewidth=1.0, alpha=0.35, zorder=0)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Accumulated Reward")
    ax.set_title("Reward accumulation over time")
    leg = ax.legend(title="Method", loc="upper right", ncol=2, frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("0.25")
    frame.set_linewidth(1.0)
    frame.set_alpha(0.9)
    frame.set_boxstyle("round,pad=0.35")
    ax.grid(True, axis="y", alpha=0.25)
    out_dir = os.path.join(RUNS_DIR, args.hyperparameters)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if SHOW_PLOTS:
        print("Showing plot...")
        fig.show()
        print("Plot shown")
    else:
        print("Saving plot...")
        fig.savefig(os.path.join(out_dir, "accumulated_reward.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.train:
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
