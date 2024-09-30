import os
import argparse
import json
from kubernetes import client, config
import time
from prom_crawler_new import *
from cluster_environment import ClusterEnvironment
from pettingzoo.test import parallel_api_test
import random
import numpy as np
import psutil


import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.DQN import DQNAgent
from agents.PPO_simple import PPO_AGENT
from agents.PPO_aware import PPO_AWARE_AGENT
from utils import *
import matplotlib.pyplot as plt
from workload_generator.WorkloadGenerator import RandomWorkloadGenerator





def main():
    print("Environment Test")
    """ALGO PARAMS"""
    print("Setting up the environment")
    print("Is cuda available ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_iterations = 2
    num_episodes = 1
    max_steps_per_episode = 10
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    skip_update = None

    print(f"Total iterations: {total_iterations}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")


    options = parse_args()
    print(options)
    
    prom_address = None
    app_name = options.app_name
    app_namespace = options.app_namespace
    file_deployment = options.file_deployment
    group_file = options.group_file
    call_host = None
    run_name = None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if options.agent == 'dqn':
        run_name = f"DQN_{timestamp}"
    elif options.agent == 'ppo_simple':
        run_name = f"PPO_SIMPLE_{timestamp}"
    elif options.agent == 'ppo_aware':
        run_name = f"PPO_AWARE_{timestamp}"
    log_dir = os.path.join("logs",)
    os.makedirs(log_dir, exist_ok=True)

    #in the same way create the checkpoint folders trying to be as specific about the configuration as possible

    checkpoint_dir = os.path.join("checkpoints", )
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_experiment_parameters(log_dir)

    results_dir = os.path.join("results", )
    os.makedirs(results_dir, exist_ok=True)





# read groups
#1 line is 1 group--> microservice1,microservice2,microservice3-[max_replicas,min_replicas....]
    groups, limits = read_groups(options.group_file)
# setup kubeconfig
    config.load_kube_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node()
    resp = v1.list_namespaced_pod(namespace=app_namespace)
    if nodes.items:
        first_node = nodes.items[0]
        for address in first_node.status.addresses:
            if address.type == "InternalIP":
                # print(f"Node Name: {first_node.metadata.name}, IP Address: {address.address}")
                break
        if options.prom_address != None:
            prom_address = options.prom_address
        else:
            prom_address = f"http://{first_node.status.addresses[0].address}:30090"

        if app_name == "teastore":
            call_host = f"http://localhost:30080/tools.descartes.teastore.webui/"
    else:
        print("No nodes found in the cluster")    


    # env setup
    env = ClusterEnvironment(groups=groups, prom_address=prom_address, kube_client= client, app_name=options.app_name, app_namespace=options.app_namespace, limits=limits, mock_observation=False)
    num_agents = len(env.possible_agents)
    observation_space = [env.observation_space(agent) for agent in env.possible_agents]
    action_space = [env.action_space(agent) for agent in env.possible_agents]
    skip_update = [False for _ in range(num_agents)]

    #deploy app and check if it is ready
    deploy_app(file_deployment, app_namespace)
    workload_generator = RandomWorkloadGenerator(app_name, call_host, log_dir)
    env.check_app_ready()


    if options.agent == 'dqn':
        agents = {agent: DQNAgent(observation_space=observation_space[agent], action_space=action_space[agent],  log_dir=f"./logs/agent_{agent}", device=device, lr=1e-3, gamma=0.99) for agent in range(num_agents)}
    elif options.agent == 'ppo_simple':
        agents = {agent: PPO_AGENT(observation_space=observation_space[agent], action_space=action_space[agent], log_dir=f"./logs/agent_{agent}", device=device, lr=3e-4, gamma=0.99, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2) for agent in range(num_agents)}
    elif options.agent == 'ppo_aware':
        timestamp = datetime.now()
        agents = {agent: PPO_AWARE_AGENT(app_name=env.app_name, timestamp_creation=timestamp,observation_space=observation_space[agent], action_space=action_space[agent], associated_agent=agent, log_dir =f"./logs/agent_{agent}" , device=device, lr=3e-4, gamma=0.99, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2) for agent in range(num_agents)}
        
        #TODO add the checkpoint loading
        # if options.use_checkpoint:
        # print('Loading model checkpoint from', options.checkpoint)
        # if os.path.exists(options.checkpoint):
        #     agent.load_checkpoint(options.checkpoint)
        # else:
        #     print('Checkpoint does not exist!')
        #     exit()

        if options.use_inference:
            print('Start RL policy serving...')
            for agent in range(num_agents):
                agents[agent].disable_update()
    else:
        print("Invalid agent type")
        exit(1)

    



    pid = os.getpid()
    print(f"PID: {pid}")
    python_process = psutil.Process(pid)



    total_timestamps = 10000
    # Initialize dictionaries
    def initialize_dicts():
        return {agent: [] for agent in range(num_agents)}

    episode_rewards = initialize_dicts()
    recent_rewards = initialize_dicts()
    iteration_rewards = initialize_dicts()
    smoothed_rewards = initialize_dicts()

    # Check app is fully deployed and ready
    for iteration in range(total_iterations):
        print(f"------------ Iteration {iteration} ------------")
        memory_usage = python_process.memory_info()[0] / 2. ** 20
        cpu_util = python_process.cpu_percent(interval=None)
        print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', cpu_util)

        states = initialize_dicts()
        actions = initialize_dicts()
        rewards = initialize_dicts()
        log_probs = initialize_dicts()

        print(f"--------- Episode 0 ---------")
        state, infos = env.reset()

        # App ready, wait for the first 15 seconds for the metrics to come in
        time.sleep(15)
        workload_thread = workload_generator.start_workload_simulation()

        episode_rewards = initialize_dicts()
        episode_states = initialize_dicts()
        episode_actions = initialize_dicts()
        episode_log_probs = initialize_dicts()

        actions_step = {agent: None for agent in range(num_agents)}
        log_probs_step = {agent: None for agent in range(num_agents)}

        for step in range(max_steps_per_episode):
            print(f"------ Step {step} ------")
            print(workload_generator.get_status())

            if options.agent == 'dqn':
                actions_step = {agent: agents[agent].select_action(state[agent], epsilon) for agent in range(num_agents)}
            elif options.agent == 'ppo_simple':
                actions_step = {agent: agents[agent].get_action_and_value(state[agent]) for agent in range(num_agents)}
            elif options.agent == 'ppo_aware':
                for agent in range(num_agents):
                    states[agent].append(state[agent])
                    episode_states[agent].append(state[agent])
                    actions_step[agent], log_probs_step[agent] = agents[agent].select_action(state[agent])
                    actions[agent].append(actions_step[agent])
                    episode_actions[agent].append(actions_step[agent])
                    log_probs[agent].append(log_probs_step[agent])
                    episode_log_probs[agent].append(log_probs_step[agent])

            print(f"---> Actions: {actions_step}")
            next_state, reward, terminated, truncated, infos = env.step(actions_step)
            env.check_app_ready()

            for agent in range(num_agents):
                episode_rewards[agent].append(reward[agent])

            if options.agent == 'dqn':
                for agent in range(num_agents):
                    agents[agent].store_transition((state[agent], actions_step[agent], reward[agent], next_state[agent], terminated[agent]))
                for agent in range(num_agents):
                    agents[agent].learn()

            state = next_state

            if all(value == True for value in terminated.values()) or all(value == True for value in truncated.values()):
                break

        for agent in range(num_agents):
            states[agent].append(next_state[agent])
            episode_states[agent].append(next_state[agent])

        

        for agent in range(num_agents):
            print(f"Agent {agent} episode rewards: {episode_rewards[agent]}")
            rewards[agent].append(episode_rewards[agent])
            print(f"Agent {agent} episode states: {episode_states[agent]}")
            if len(recent_rewards[agent]) < MAX_NUM_REWARDS_TO_CHECK:
                recent_rewards[agent].append(np.sum(episode_rewards[agent]))
                print(f"Recent rewards for agent {agent}: {np.sum(episode_rewards[agent])}")
            else:
                recent_rewards[agent].pop(0)
                recent_rewards[agent].append(np.sum(episode_rewards[agent]))
                print(f"Recent rewards for agent {agent}: {np.sum(episode_rewards[agent])}")

            avg_reward = np.mean(recent_rewards[agent])
            std_reward = np.std(recent_rewards[agent])

            if skip_update[agent]:
                print(f"Checking if policy re-training is needed for agent {agent}")
                if avg_reward < REWARD_AVG_THRESHOLD and std_reward > REWARD_STD_THRESHOLD:
                    print(f"Policy re-training needed for agent {agent}")
                    skip_update[agent] = False

            if not skip_update[agent]:
                print(f"Updating policy for agent {agent}")
                if avg_reward >= REWARD_AVG_THRESHOLD and std_reward < REWARD_STD_THRESHOLD:
                    print(f"Training completed for {agent}")
                    print(f"Average Reward: {avg_reward}, Std Dev Reward: {std_reward}")
                    skip_update[agent] = True

        if options.agent == 'dqn':
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        print(f"Episode 0 completed")
        workload_generator.set_terminated(True)
        time.sleep(1)

        print(f"------------ Iteration {iteration} completed, Results ------------")

        if options.agent in ['ppo_simple', 'ppo_aware']:
            all_rewards = initialize_dicts()
            overall_average_rewards = np.mean([reward for episode_rewards in rewards.values() for reward in episode_rewards])
            print(' --> Overall average rewards across all agents:', np.round(overall_average_rewards, decimals=3))

            for agent in range(num_agents):
                average_rewards = np.mean([reward for reward in rewards[agent]])
                iteration_rewards[agent].append(average_rewards)
                smoothed_rewards[agent].append(np.mean(iteration_rewards[agent][-10:]))
                print(f' --> Average rewards for agent {agent}: {np.round(average_rewards, decimals=3)}, Moving average: {np.round(np.mean(iteration_rewards[agent][-10:]), decimals=3)}')

                if SAVE_TO_FILE:
                    all_rewards[agent] = [reward for reward_ep in rewards[agent] for reward in reward_ep]
                    agents[agent].save_trajectories(iteration=iteration, states=states[agent], actions=actions[agent], rewards=all_rewards[agent], log_probs=log_probs[agent])
                    print(f'Trajectory for {agent} saved to file')

        if iteration % 1 == 0 and iteration != 0:
            for agent in range(num_agents):
                agents[agent].save_checkpoint(iteration, timestamp_creation=timestamp, model=options.agent, agent=agent)
    for agent in range(num_agents):     
        if PLOT_FIG:
            visualization(iteration_rewards[agent], smoothed_rewards[agent], app_name, agent, options.agent, timestamp, results_dir)

        
        if SAVE_TO_FILE:
            save_results(iteration_rewards[agent], smoothed_rewards[agent], app_name, agent, options.agent, timestamp, results_dir)


    env.close()
            


def read_groups(file):
    print("Trying to read group")
    dict_groups = {}
    dict_limits = {}
    i = 0
    # check if file exists
    if os.path.exists(file):
        with open(file, 'r') as file:
            print("Reading groups")
            for line in file:
                if not line.startswith('#'):
                    parts = line.strip().split('\\')
                    microservices = parts[0].strip().split(',')
                    limits = parts[1].strip().split(',')
                    print(limits)
                    dict_groups[i] = microservices
                    dict_limits[i] = {
                        'max_replicas': int(limits[1]),
                        'min_replicas': int(limits[0]),
                    }
                    i += 1

        print("Groups read")
        return dict_groups, dict_limits
    else:
        print("Group file does not exist")


def parse_args():

    parser = argparse.ArgumentParser()

    #my arguments
    parser.add_argument('--app_name', type=str, default='teastore')  # name of the app to control
    parser.add_argument('--file_deployment', type=str, default="teastore-1c-5gib.yaml")
    parser.add_argument('--app_namespace', type=str, default='default')  # namespace of the app
    parser.add_argument('--use_inference', action='store_true')  # True for skipping RL training, default False
    parser.add_argument('--use_checkpoint',action='store_true')  # True for loading from a model checkpoint, default False
    parser.add_argument('--checkpoint', type=str)  # path to the checkpoint file
    parser.add_argument('--group_file', type=str, default="./groups.txt")  # number of groups in the app
    parser.add_argument('--prom_address', type=str)
    # parser.add_argument('--bootstrapping', action='store_true')  # True for bootstrapping RL training (offline), default False
    parser.set_defaults(use_inference=False)
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--agent', type=str, default='ppo_aware')  # name of the agent to use



    
    args = parser.parse_args()
    return args

def visualization(iteration_rewards, smoothed_rewards ,app_name ,agent, model, timestamp, dir):
    timestamp = int(datetime.timestamp(timestamp))
    graphics_dir = f'./{dir}/{app_name}/{str(timestamp)}_{model}/agent{str(agent)}/graphics'
    os.makedirs(graphics_dir, exist_ok=True)
    #rewards

    plt.plot(iteration_rewards, color='steelblue', alpha=0.3)
    plt.plot(smoothed_rewards, color='steelblue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.tight_layout()
    if not SAVE_FIG:
        plt.show()
    else:
        plt.savefig(f'{graphics_dir}/rewards.svg')


def save_results(iteration_rewards, smoothed_rewards, app_name ,agent, model, timestamp, dir):

    timestamp = int(datetime.timestamp(timestamp))
    results_dir = f'./{dir}/{app_name}/{str(timestamp)}_{model}/agent{str(agent)}/textual'
    os.makedirs(results_dir, exist_ok=True)

    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/rewards.txt', 'w') as f:
        for i in range(len(iteration_rewards)):
            f.write(f'Iteration{i},{iteration_rewards[i]},{smoothed_rewards[i]}\n')


def deploy_app(file_deployment, app_namespace):
    #to customize. DEFAULT TEASTORE
    print("Deploying app")
    os.system(f"kubectl apply -f ./apps/{file_deployment} -n {app_namespace}")
    print("App deployed")

def save_experiment_parameters(log_dir):
    pass


if __name__ == '__main__':
    main()





# fmt: off
    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    #     help="the name of this experiment")
    # parser.add_argument("--seed", type=int, default=1,
    #     help="seed of the experiment")
    # parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, cuda will be enabled by default")
    # parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="if toggled, this experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to capture videos of the agent performances (check out `videos` folder)")

    # # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="cluster_environment",
    #     help="the id of the environment")
    # parser.add_argument("--total-timesteps", type=int, default=12000,  # CleanRL default: 2000000
    #     help="total timesteps of the experiments")
    # parser.add_argument("--learning-rate", type=float, default=2.5e-4,
    #     help="the learning rate of the optimizer")
    # parser.add_argument("--num-envs", type=int, default=16,
    #     help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=128,
    #     help="the number of steps to run in each environment per policy rollout")
    # parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggle learning rate annealing for policy and value networks")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #     help="the discount factor gamma")
    # parser.add_argument("--gae-lambda", type=float, default=0.95,
    #     help="the lambda for the general advantage estimation")
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="the number of mini-batches")
    # parser.add_argument("--update-epochs", type=int, default=4,
    #     help="the K epochs to update the policy")
    # parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles advantages normalization")
    # parser.add_argument("--clip-coef", type=float, default=0.1,
    #     help="the surrogate clipping coefficient")
    # parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    # parser.add_argument("--ent-coef", type=float, default=0.01,
    #     help="coefficient of the entropy")
    # parser.add_argument("--vf-coef", type=float, default=0.5,
    #     help="coefficient of the value function")
    # parser.add_argument("--max-grad-norm", type=float, default=0.5,
    #     help="the maximum norm for the gradient clipping")
    # parser.add_argument("--target-kl", type=float, default=None,
    #     help="the target KL divergence threshold")

    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

