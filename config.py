FRAME_SKIP = 4
STACK_SIZE = 4
IMAGE_SIZE = 84

GAMMA = 0.9
LEARNING_RATE = 0.00025
BATCH_SIZE = 32

NUM_EPISODES = 100 # temporary for testing, change to 1000-10000 for full training
MAX_STEPS = 1000 # temporary for testing, change to 10000-100000 for full training

SAVE_EVERY_N_STEPS = 10
VIDEO_FPS = 10

EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.95

CHECKPOINT_FILE = "mario_dqn.pth"

LOGS_DIR = "logs"
TRAINING_LOG_FILE = "logs/training.csv"

TARGET_UPDATE_EVERY = 5
