from pathlib import Path
import csv

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_agent.agent import MarioAgent
from mario_agent.wrappers import (
    create_initial_stack,
    update_stack,
    step_with_frame_skip,
)

from config import (
    STACK_SIZE,
    IMAGE_SIZE,
    NUM_EPISODES,
    MAX_STEPS,
    BATCH_SIZE,
    FRAME_SKIP,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    CHECKPOINT_FILE,
    LOGS_DIR,
    TRAINING_LOG_FILE,
    TARGET_UPDATE_EVERY
)

def main():

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    state_shape = (STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    num_actions = env.action_space.n

    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions
    )

    num_episodes = NUM_EPISODES
    max_steps = MAX_STEPS
    batch_size = BATCH_SIZE
    epsilon = EPSILON_START

    Path(LOGS_DIR).mkdir(exist_ok=True)

    with open(TRAINING_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "steps",
            "reward",
            "last_loss",
            "epsilon",
            "memory_length"
        ])

    for episode in range(num_episodes):

        state = env.reset().copy()
        stacked_state = create_initial_stack(state)
        episode_reward = 0
        loss = None

        for step in range(max_steps):
            
            action = agent.select_action(
                stacked_state,
                epsilon=epsilon
            )

            next_state, reward, done, info = step_with_frame_skip(
                env,
                action=action,
                frame_skip=FRAME_SKIP
            )

            next_stacked_state = update_stack(
                stacked_state,
                next_state.copy()
            )

            agent.remember(
                stacked_state,
                action,
                reward,
                next_stacked_state,
                done
            )

            loss = agent.learn(batch_size=batch_size)

            stacked_state = next_stacked_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}")
        print("Steps:", step + 1)
        print("Episode reward:", episode_reward)
        print("Memory length:", len(agent.memory))
        print("Last loss:", loss if loss is not None else "Not enough memory yet")
        print("Epsilon:", epsilon)
        print("-" * 40)

        # log row to CSV

        if (episode + 1) % TARGET_UPDATE_EVERY == 0:
            agent.update_target_model()
            print("Target model updated")

        agent.save(CHECKPOINT_FILE)


        with open(TRAINING_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                step + 1,
                episode_reward,
                loss if loss is not None else "",
                epsilon,
                len(agent.memory)
            ])

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


    env.close()


if __name__ == "__main__":
    main()