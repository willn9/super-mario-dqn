import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from mario_agent.wrappers import preprocess_frame, create_initial_stack, update_stack, step_with_frame_skip


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    state = env.reset()

    processed_state = preprocess_frame(state)

    stacked_state = create_initial_stack(state)

    print("Stacked shape:", stacked_state.shape)
    print("Stacked dtype:", stacked_state.dtype)
    print("Stacked min/max:", stacked_state.min(), stacked_state.max())

    print("Processed shape:", processed_state.shape)
    print("Processed dtype:", processed_state.dtype)
    print("Processed min/max:", processed_state.min(), processed_state.max())

    print("Environment created successfully")
    print("Observation shape:", state.shape)
    print("Action space:", env.action_space)
    print("Number of actions:", env.action_space.n)

    next_state, reward, done, info = step_with_frame_skip(env, action=1, frame_skip=4)

    updated_stack = update_stack(stacked_state, next_state)

    print("Updated stack shape:", updated_stack.shape)
    print("Updated stack dtype:", updated_stack.dtype)
    print("Updated stack min/max:", updated_stack.min(), updated_stack.max())

    print("Step worked")
    print("Next observation shape:", next_state.shape)
    print("Reward:", reward)
    print("Done:", done)
    print("Info keys:", info.keys())

    env.close()


if __name__ == "__main__":
    main()