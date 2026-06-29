import numpy as np
from torch.nn.modules import loss

from mario_agent.agent import MarioAgent


def main():

    state_shape = (4, 84, 84)
    num_actions = 7

    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions
    )

    dummy_state = np.zeros(
        (4, 84, 84),
        dtype=np.float32
    )

    print("Greedy action:")
    print(agent.select_action(dummy_state, epsilon=0.0))

    print("\nRandom actions:")
    for _ in range(10):
        print(agent.select_action(dummy_state, epsilon=1.0))

    
    next_state = np.ones(
    (4, 84, 84),
    dtype=np.float32
    )

    agent.remember(
        state=dummy_state,
        action=1,
        reward=0.5,
        next_state=next_state,
        done=False
    )

    print("\nMemory length:", len(agent.memory))

    print("Can learn with batch_size=1:", agent.can_learn(batch_size=1))
    print("Can learn with batch_size=32:", agent.can_learn(batch_size=32))

    loss = agent.learn(batch_size=1)
    print("Learn returned:", loss)


if __name__ == "__main__":
    main()