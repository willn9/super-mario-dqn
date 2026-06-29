import torch

from mario_agent.model import MarioDQN


def main():
    input_shape = (4, 84, 84)
    num_actions = 7
    

    model = MarioDQN(input_shape=input_shape, num_actions=num_actions)

    dummy_state = torch.zeros(1, 4, 84, 84)

    q_values = model(dummy_state)

    print(model)
    print("Input shape:", dummy_state.shape)
    print("Output shape:", q_values.shape)
    print("Q-values:", q_values)


if __name__ == "__main__":
    main()