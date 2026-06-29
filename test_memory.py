import numpy as np

from mario_agent.memory import ReplayMemory


def main():
    memory = ReplayMemory(capacity=100)

    state = np.zeros((4, 84, 84), dtype=np.float32)
    next_state = np.ones((4, 84, 84), dtype=np.float32)

    memory.push(state, 1, 0.5, next_state, False)

    print("Memory length:", len(memory))

    batch = memory.sample(1)
    print("Sample length:", len(batch))

    sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done = batch[0]

    print("Sampled state shape:", sampled_state.shape)
    print("Sampled action:", sampled_action)
    print("Sampled reward:", sampled_reward)
    print("Sampled next_state shape:", sampled_next_state.shape)
    print("Sampled done:", sampled_done)


if __name__ == "__main__":
    main()