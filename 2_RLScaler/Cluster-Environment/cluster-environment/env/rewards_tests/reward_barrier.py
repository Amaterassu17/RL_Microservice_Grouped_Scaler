import math
import matplotlib.pyplot as plt
import numpy as np

# Constants for thresholds
HIGH_UTIL_THRESHOLD = 0.8
MID_HIGH_UTIL_THRESHOLD = 0.6
MID_LOW_UTIL_THRESHOLD = 0.4
LOW_UTIL_THRESHOLD = 0.2
MAX_REPLICAS = 5

# Function to calculate the reward for scaling action considering before and after scaling utilization
def calculate_reward_with_before_after(avg_utilization, last_avg_utilization, scaling_action, num_replicas, last_num_replicas):
    utilization_diff = avg_utilization - last_avg_utilization
    reward = 0

    if scaling_action > 0:  # Scaling up
        if last_avg_utilization > HIGH_UTIL_THRESHOLD:
            reward = math.exp(-utilization_diff)  # Exponential reward for reducing high utilization
        elif MID_HIGH_UTIL_THRESHOLD < last_avg_utilization <= HIGH_UTIL_THRESHOLD:
            reward = 1 - utilization_diff  # Linear reward for reducing moderate high utilization
        elif MID_LOW_UTIL_THRESHOLD < last_avg_utilization <= MID_HIGH_UTIL_THRESHOLD:
            reward = -0.5 * utilization_diff  # Penalty for unnecessary scaling
        else:
            reward = -math.exp(utilization_diff)  # Large penalty for scaling when not needed

    elif scaling_action < 0:  # Scaling down
        if last_avg_utilization < LOW_UTIL_THRESHOLD:
            reward = math.exp(utilization_diff)  # Exponential reward for increasing low utilization
        elif LOW_UTIL_THRESHOLD <= last_avg_utilization < MID_LOW_UTIL_THRESHOLD:
            reward = utilization_diff  # Linear reward for increasing moderate low utilization
        elif MID_LOW_UTIL_THRESHOLD <= last_avg_utilization < MID_HIGH_UTIL_THRESHOLD:
            reward = -0.5 * utilization_diff  # Penalty for unnecessary scaling
        else:
            reward = -math.exp(-utilization_diff)  # Large penalty for scaling down when not needed

    else:  # No scaling action
        if MID_LOW_UTIL_THRESHOLD < last_avg_utilization <= MID_HIGH_UTIL_THRESHOLD:
            reward = 0.5  # Reward for maintaining stable state
        else:
            reward = -0.5  # Penalty for not scaling when needed

    return reward

# Function to plot the rewards in 3D
def plot_rewards_3d():
    avg_utilizations = np.linspace(0, 1.2, 100)
    last_avg_utilizations = np.linspace(0, 1.2, 100)
    num_replicas = 3  # Current number of replicas
    last_num_replicas = 2  # Last number of replicas
    scaling_action = num_replicas - last_num_replicas

    avg_utilizations_grid, last_avg_utilizations_grid = np.meshgrid(avg_utilizations, last_avg_utilizations)
    rewards_grid = np.zeros_like(avg_utilizations_grid)

    for i in range(avg_utilizations_grid.shape[0]):
        for j in range(avg_utilizations_grid.shape[1]):
            avg_utilization = avg_utilizations_grid[i, j]
            last_avg_utilization = last_avg_utilizations_grid[i, j]
            rewards_grid[i, j] = calculate_reward_with_before_after(avg_utilization, last_avg_utilization, scaling_action, num_replicas, last_num_replicas)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(avg_utilizations_grid, last_avg_utilizations_grid, rewards_grid, cmap='viridis')

    ax.set_xlabel('Current Average Utilization')
    ax.set_ylabel('Last Average Utilization')
    ax.set_zlabel('Reward')
    ax.set_title('Reward Function for Scaling Actions in 3D')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Plot the rewards in 3D
plot_rewards_3d()
