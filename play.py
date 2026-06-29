from pathlib import Path
import cv2

import json
from datetime import datetime

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
    FRAME_SKIP,
    SAVE_EVERY_N_STEPS,
    VIDEO_FPS,
    CHECKPOINT_FILE
)

def create_video_from_frames(frames_dir, output_path, fps=10):
    frame_paths = sorted(frames_dir.glob("*.png"))

    if not frame_paths:
        print("No frames found; video was not created.")
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        video_writer.write(frame)

    video_writer.release()

    print(f"Video saved to: {output_path}")


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    state_shape = (STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    num_actions = env.action_space.n

    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions
    )

    agent.load(CHECKPOINT_FILE)
    agent.model.eval()

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    run_id = len(list(runs_dir.glob("run_*"))) + 1
    run_dir = runs_dir / f"run_{run_id:04d}"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to: {frames_dir}")

    state = env.reset().copy()
    stacked_state = create_initial_stack(state)

    episode_reward = 0

    for step in range(1000):
        action = agent.select_action(
            stacked_state,
            epsilon=0.0
        )

        next_state, reward, done, info = step_with_frame_skip(
            env,
            action=action,
            frame_skip=FRAME_SKIP
        )

        stacked_state = update_stack(
            stacked_state,
            next_state.copy()
        )

        episode_reward += reward

        if step % SAVE_EVERY_N_STEPS == 0:
            frame = next_state.copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            text = (
                f"Step: {step} | "
                f"Reward: {episode_reward:.1f} | "
                f"X: {info['x_pos']} | "
                f"Action: {action}"
            )

            cv2.putText(
                frame_bgr,
                text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            frame_path = frames_dir / f"frame_{step:04d}_x{info['x_pos']:04d}.png"
            cv2.imwrite(str(frame_path), frame_bgr)

        if step % 100 == 0:
            print(
                f"Step {step} | "
                f"Reward: {episode_reward:.1f} | "
                f"X Position: {info['x_pos']}"
            )

        if done:
            break

    print("\nEpisode finished \n")
    print("Steps:", step + 1)
    print("Reward:", episode_reward)
    print("Final X Position:", info["x_pos"])
    print(f"Frames saved to: {frames_dir}")

    # Create a video from the saved frames
    video_path = run_dir / "run.mp4"
    create_video_from_frames(
    frames_dir,
    video_path,
    fps=VIDEO_FPS
    )
    
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": CHECKPOINT_FILE,
        "steps": int(step + 1),
        "reward": float(episode_reward),
        "final_x_position": int(info["x_pos"]),
        "save_every_n_steps": int(SAVE_EVERY_N_STEPS),
        "video_fps": int(VIDEO_FPS),
        "created_at": datetime.now().isoformat(),
    }

    summary_path = run_dir / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to: {summary_path}")    

    env.close()


if __name__ == "__main__":
    main()