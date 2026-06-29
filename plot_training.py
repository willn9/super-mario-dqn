import pandas as pd
import matplotlib.pyplot as plt

from config import TRAINING_LOG_FILE


def main():
    df = pd.read_csv(TRAINING_LOG_FILE)

    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.savefig("logs/reward_vs_episode.png")
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["last_loss"])
    plt.xlabel("Episode")
    plt.ylabel("Last Loss")
    plt.title("Loss vs Episode")
    plt.savefig("logs/loss_vs_episode.png")
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["epsilon"])
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon vs Episode")
    plt.savefig("logs/epsilon_vs_episode.png")
    plt.close()

    print("Plots saved to logs/")


if __name__ == "__main__":
    main()