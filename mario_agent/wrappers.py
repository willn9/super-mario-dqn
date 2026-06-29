import cv2
import numpy as np
from config import IMAGE_SIZE, STACK_SIZE, FRAME_SKIP


def preprocess_frame(frame):
    """
    Convert a raw Mario RGB frame into a normalized 84x84 grayscale frame.

    Input:
        frame: np.ndarray with shape (240, 256, 3), dtype uint8

    Output:
        processed: np.ndarray with shape (84, 84), dtype float32, values 0-1
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized / 255.0

    return normalized.astype(np.float32)

def create_initial_stack(frame, stack_size=STACK_SIZE):
    """
    Create the initial state stack by repeating the first processed frame.
    now we are using config.py to set the stack size and image size

    Output shape example:
        (stack_size, 84, 84)
    """
    processed = preprocess_frame(frame)
    stacked = np.stack([processed] * stack_size, axis=0)

    return stacked.astype(np.float32)

def update_stack(stacked_frames, new_frame):
    """
    Drop the oldest frame and append the newest processed frame.

    Input:
        stacked_frames: np.ndarray with shape (4, 84, 84)
        new_frame: raw RGB frame from Mario

    Output:
        updated_stack: np.ndarray with shape (4, 84, 84)
    """
    processed = preprocess_frame(new_frame)

    updated_stack = np.concatenate(
        (stacked_frames[1:], processed[np.newaxis, :, :]),
        axis=0
    )

    return updated_stack.astype(np.float32)


def step_with_frame_skip(env, action, frame_skip=FRAME_SKIP):
    """
    Repeat the same action for several frames.

    Returns:
        final_frame, total_reward, done, info
    """
    total_reward = 0.0
    done = False
    info = {}

    for _ in range(frame_skip):
        frame, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    return frame, total_reward, done, info