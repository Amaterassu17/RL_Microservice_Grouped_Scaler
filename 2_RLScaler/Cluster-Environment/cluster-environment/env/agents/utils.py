#DQN UTIL


#SIMPLE PPO UTIL



#AWARE PPO UTIL

"""
Hyperparameters
"""
TOTAL_ITERATIONS = 500
EPISODES_PER_ITERATION = 50
EPISODE_LENGTH = 10

DISCOUNT = 0.99
HIDDEN_SIZE = 64
LR = 3e-4  # 5e-3 5e-6
SGD_EPOCHS = 5
MINI_BATCH_SIZE = 5
CLIP = 0.2
ENTROPY_COEFFICIENT = 0.01  # 0.001
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03

FLAG_CONTINUOUS_ACTION = False

MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS
MAX_NUM_REWARDS_TO_CHECK = 10

# thresholds for retraining monitoring
REWARD_AVG_THRESHOLD = 100
REWARD_STD_THRESHOLD = 10

ILLEGAL_PENALTY = -1


"""
Logging and Checkpointing
"""
CHECKPOINT_DIR = './checkpoints/'
LOG_DIR = './logs/'
PLOT_FIG = True
SAVE_FIG = True
SAVE_TO_FILE = True
DATA_PATH = 'data.csv'
SAVE_MODEL = True


"""
Resource Scaling Constraints
"""
MIN_INSTANCES = 1
MAX_INSTANCES = 20    # determined by the cluster capacity
MIN_CPU_LIMIT = 128   # millicore
MAX_CPU_LIMIT = 2048  # millicore
MIN_MEMORY_LIMIT = 256   # MiB
MAX_MEMORY_LIMIT = 3072  # MiB

LOWER_BOUND_UTIL = 0.7
UPPER_BOUND_UTIL = 0.9
