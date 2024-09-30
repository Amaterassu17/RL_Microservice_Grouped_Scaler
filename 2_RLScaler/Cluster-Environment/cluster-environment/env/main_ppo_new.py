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
import wandb
from agents.PPO_simple import PPO_AGENT
from agents.PPO_aware import PPO_AWARE_AGENT
from utils import *
import matplotlib.pyplot as plt
import datetime
from workload_generator.WorkloadGenerator import RandomWorkloadGenerator

class RLParams:
    def __init__(self,
                # Arguments related to the application
                app_name: str = 'teastore',
                file_deployment: str = "teastore-1c-5gib.yaml",
                app_namespace: str = 'default',
                use_inference: bool = False,
                use_checkpoint: bool = False,
                checkpoint: str = None,
                group_file: str = "./groups.txt",
                prom_address: str = None,
                agent: str = 'ppo_aware',
                timestamp: datetime = int(datetime.datetime.now().timestamp()),
                save_to_file: bool = True,
                plot_fig: bool = True,

                # PPO hyperparameters
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,               # Discount factor
                 k_epochs: int = 5,                 # Update policy for K epochs
                 eps_clip: float = 0.2,             # Clipping ratio for PPO
                 vf_coef: float = 0.5,              # Value function coefficient
                 ent_coef: float = 0.01,            # Entropy coefficient
                 max_grad_norm: float = 0.5,        # Max norm for gradient clipping
                 critic_loss_discount: float = 0.05,  # Discount factor for critic loss
                 minibatch_size: int = 5,           # Mini-batch size
                 max_num_rewards_to_check = 10,     # Number of rewards to check
                 gae_lam: float = 0.95,             # Generalized Advantage Estimation lambda



                 # Training parameters
                 total_iterations: int = 1000,
                 num_episodes: int = 1000,
                 max_steps_per_episode: int = 200,          # Max timesteps in one episode
                 update_timestep: int = 4000,       # Timestep threshold for updating policy
                 save_interval: int = 50, 
                 model = 'ppo_aware',
                 reward_type = 'aware',
                 seed: int = 42,
                 num_tries_cluster_stable: int = 8,

                 # Policy network parameters
                 hidden_size: int = 64,             # Number of hidden units per layer
                 num_layers: int = 2,               # Number of hidden layers
    
                 # Retraining monitoring thresholds
                    reward_avg_threshold: int = 100,  # Average reward threshold for retraining
                    reward_std_threshold: int = 10,   # Standard deviation threshold for retraining

                 #
    
                 # Machine parameters
                 
                 
    
    ):
        
        self.app_name = app_name
        self.file_deployment = file_deployment
        self.app_namespace = app_namespace
        self.use_inference = use_inference
        self.use_checkpoint = use_checkpoint
        self.checkpoint = checkpoint
        self.group_file = group_file
        self.prom_address = prom_address
        self.agent = agent
        self.timestamp = timestamp
        self.save_to_file = save_to_file
        self.plot_fig = plot_fig

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.critic_loss_discount = critic_loss_discount
        self.minibatch_size = minibatch_size
        self.max_num_rewards_to_check = max_num_rewards_to_check
        self.gae_lam = gae_lam

        self.total_iterations = total_iterations
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_timestep = update_timestep
        self.save_interval = save_interval
        self.num_agents = 1
        self.skip_update = 0
        self.model = model
        self.reward_type = reward_type
        self.seed = seed
        self.num_tries_cluster_stable = num_tries_cluster_stable

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.reward_avg_threshold = reward_avg_threshold
        self.reward_std_threshold = reward_std_threshold

        self.num_nodes = 1
        self.cpu_name = "cpu"
        self.num_cores = 4
        self.memory = 8
        self.gpu_name = "gpu"
        self.num_gpus = 1
        self.hostnames = []
    

    def __str__(self):
        return f"Total iterations: {self.total_iterations}, Number of episodes: {self.num_episodes}, Max steps per episode: {self.max_steps_per_episode}, Epsilon: {self.epsilon}, Epsilon decay: {self.epsilon_decay}, Epsilon min: {self.epsilon_min}, Skip update: {self.skip_update}"

    def display(self):
        config_dict = self.__dict__
        for key, value in config_dict.items():
            print(f"{key}: {value}")


# Helper function to filter out arguments that are different from defaults
def filter_args(defaults, args):
    return {key: value for key, value in vars(args).items() if value != getattr(defaults, key)}



def main():

    # np.seterr(all='raise')
    # torch.autograd.set_detect_anomaly(True)
    # torch.set_printoptions(profile="full")
    
    
        
    # Initialize dictionaries
    def initialize_dicts():
        return {agent: [] for agent in range(num_agents)}
    
    print("Setting up the environment")
    print("Is cuda available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    options = parse_args()
    print(options)

    default_rl_params = RLParams()
    updated_args = filter_args(default_rl_params, options)


    rl_params = RLParams(**updated_args)

    rl_params.display()
    
    #retrieve host name
    hostname = os.uname()[1]
    

    #################################################################

    rl_params.total_iterations = 300
    rl_params.max_steps_per_episode = 60
    rl_params.num_agents = 2
    rl_params.reward_type = 'simple3'
    rl_params.skip_update = [False, False]
    rl_params.seed=561
    rl_params.minibatch_size = 5
    rl_params.num_tries_cluster_stable = 5

    #TEMPORARY REPLACEMENT FOR THE ARGUMENTS UNTIL I HAVE A SCRIPT
    
    print(f"Total iterations: {rl_params.total_iterations}")
    print(f"Number of episodes: {rl_params.num_episodes}")
    print(f"Max steps per episode: {rl_params.max_steps_per_episode}")
    
    app_name = rl_params.app_name
    app_namespace = rl_params.app_namespace
    file_deployment = rl_params.file_deployment
    timestamp = rl_params.timestamp
    
    log_dir = os.path.join("logs",)
    checkpoint_dir = os.path.join("checkpoints", )
    results_dir = os.path.join("results", )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_experiment_parameters(log_dir, rl_params.app_name, timestamp, rl_params)

      
    #initialize tensorboard
    tensor_runs_dir = os.path.join("tensorboard_logs", app_name, str(timestamp))
    writer = SummaryWriter(log_dir=tensor_runs_dir)
    
    print(rl_params.group_file)
    groups, limits = read_groups(rl_params.group_file)
    
    wandb.init(project='rl_grouped_scaling_09_08_2024', config={
    'project_name': 'rl_' + str(timestamp),
    'total_iterations': rl_params.total_iterations,
    'num_episodes': rl_params.num_episodes,
    'max_steps_per_episode': rl_params.max_steps_per_episode,
    'num_agents': rl_params.num_agents,
    'skip_update': rl_params.skip_update,
    'learning_rate': rl_params.learning_rate,
    'gamma': rl_params.gamma,
    'k_epochs': rl_params.k_epochs,
    'eps_clip': rl_params.eps_clip,
    'ent_coef': rl_params.ent_coef,
    'vf_coef': rl_params.vf_coef,
    'max_grad_norm': rl_params.max_grad_norm,
    'critic_loss_discount': rl_params.critic_loss_discount,
    'minibatch_size': rl_params.minibatch_size,
    'max_num_rewards_to_check': rl_params.max_num_rewards_to_check,
    'gae_lam': rl_params.gae_lam,
    'app_name': rl_params.app_name,
    'app_namespace': rl_params.app_namespace,
    'reward' : rl_params.reward_type,
    'timestamp' : timestamp,
    'hidden_size': rl_params.hidden_size,
    'num_layers': rl_params.num_layers,
    'seed': rl_params.seed,
    'host': hostname,
    'num_tries_cluster_stable': rl_params.num_tries_cluster_stable,
    'groups': json.dumps(groups),
})

    
    hyperparameters = {
    'learning_rate': rl_params.learning_rate,
    'gamma': rl_params.gamma,
    'k_epochs': rl_params.k_epochs,
    'eps_clip': rl_params.eps_clip,
    'ent_coef': rl_params.ent_coef,
    'vf_coef': rl_params.vf_coef,
    'max_grad_norm': rl_params.max_grad_norm,
    'critic_loss_discount': rl_params.critic_loss_discount,
    'minibatch_size': rl_params.minibatch_size,
    'max_num_rewards_to_check': rl_params.max_num_rewards_to_check,
    'gae_lam': rl_params.gae_lam,
    'num_tries_cluster_stable': rl_params.num_tries_cluster_stable,
    'hidden_size': rl_params.hidden_size,
    'num_layers': rl_params.num_layers,
    'seed': rl_params.seed,
    'reward': rl_params.reward_type,
    # Add any other hyperparameters here
}

    writer.add_text('Hyperparameters', 
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparameters.items()])))
        
    wandb.config.update(hyperparameters)

    #seeding
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)  
    torch.backends.cudnn.deterministic = True


    # read groups
    #1 line is 1 group--> microservice1,microservice2,microservice3-[max_replicas,min_replicas....]
    

    rl_params.num_agents = len(groups)
    rl_params.skip_update = [False for _ in range(rl_params.num_agents)]


    # setup kubeconfig
    config.load_kube_config()
    v1 = client.CoreV1Api()
    prom_address, call_host = get_addresses(v1.list_node(), options, rl_params)


    # env setup
    env = ClusterEnvironment(rl_params=rl_params, groups=groups, prom_address=prom_address, kube_client= client, limits=limits)
    num_agents = len(env.possible_agents)
    observation_space = [env.observation_space(agent) for agent in env.possible_agents]
    action_space = [env.action_space(agent) for agent in env.possible_agents]
    #deploy app and check if it is ready
    undeploy_app(file_deployment, app_namespace)
    deploy_app(file_deployment, app_namespace)
    workload_generator = RandomWorkloadGenerator(app_name, call_host, log_dir)
    env.check_app_ready()
    skip_update = [False for _ in range(num_agents)]

    # Initialize agents
    agents = {agent: PPO_AWARE_AGENT(rl_params=rl_params, observation_space=observation_space[agent], action_space=action_space[agent], associated_agent=agent , device=device, writer=writer) for agent in range(num_agents)}
    
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
    
    pid = os.getpid()
    print(f"PID: {pid}")
    python_process = psutil.Process(pid)

    episode_rewards = initialize_dicts()
    recent_rewards = initialize_dicts()
    iteration_rewards = initialize_dicts()
    smoothed_rewards = initialize_dicts()

    new_learning_rate = rl_params.learning_rate
    cumulative_rewards = {agent: 0.0 for agent in range(num_agents)}
    overall_cumulative_rewards = 0.0
    total_steps = 0

    # Check app is fully deployed and ready
    for iteration in range(rl_params.total_iterations):
        print(f"------------ Iteration {iteration} ------------")
        memory_usage = python_process.memory_info()[0] / 2. ** 20
        cpu_util = python_process.cpu_percent(interval=None)
        writer.add_scalar('System/Memory Usage (MB)', memory_usage, iteration)
        writer.add_scalar('System/CPU Utilization (%)', cpu_util, iteration)
        wandb.log({'System/Memory Usage (MB)': memory_usage, 'iteration': iteration})
        wandb.log({'System/CPU Utilization (%)': cpu_util, 'iteration': iteration})
 
        print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', cpu_util)

        states = initialize_dicts()
        actions = initialize_dicts()
        rewards = initialize_dicts()
        log_probs = initialize_dicts()

        
        state, infos = env.reset()

        # App ready, wait for the first 15 seconds for the metrics to come in
        time.sleep(30)
        workload_thread = workload_generator.start_workload_simulation()

        episode_rewards = initialize_dicts()
        episode_states = initialize_dicts()
        episode_actions = initialize_dicts()
        episode_log_probs = initialize_dicts()

        actions_step = {agent: None for agent in range(num_agents)}
        log_probs_step = {agent: None for agent in range(num_agents)}
        

        for step in range(rl_params.max_steps_per_episode):
            print(f"------ Step {step} ------")
            print(workload_generator.get_status())

            for agent in range(num_agents):
                # print(f"Agent {agent} state: {state[agent]}")
                states[agent].append(state[agent])
                # print("States after append" ,states)
                episode_states[agent].append(state[agent])
                actions_step[agent], log_probs_step[agent] = agents[agent].select_action(state[agent])
                actions[agent].append(actions_step[agent])
                episode_actions[agent].append(actions_step[agent])
                log_probs[agent].append(log_probs_step[agent])
                episode_log_probs[agent].append(log_probs_step[agent])

            print(f"---> Actions: {actions_step}")
            next_state, reward, terminated, truncated, infos = env.step(actions_step)
            # print(f"Next state: {next_state}")
            env.check_app_ready()

            for agent in range(num_agents):
                episode_rewards[agent].append(reward[agent])
                cumulative_rewards[agent] += reward[agent]
                wandb.log({f"Agent_{agent}_Steps/1Reward": reward[agent], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/1Cumulative_Reward": cumulative_rewards[agent], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/3Avg_Cpu_Util": state[agent][0][0], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/3Avg_Mem_Util": state[agent][0][1], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/3SLO_Latency50": state[agent][2][0], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/3SLO_Latency95": state[agent][2][1], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/2Num_Replica_Chosen": infos[agent]['num_replicas'], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/2Old_Best_Replica": infos[agent]['old_best_replica'], "step": total_steps})
                wandb.log({f"Agent_{agent}_Steps/2Difference_replicas": infos[agent]['num_replicas'] - infos[agent]['old_best_replica'], "step": total_steps})

            state = next_state
            total_steps = total_steps + 1
            if all(value == True for value in terminated.values()) or all(value == True for value in truncated.values()):
                break

        
        for agent in range(num_agents):
            # print(f"Agent {agent} episode rewards: {episode_rewards[agent]}")
            rewards[agent].append(episode_rewards[agent])
            # print(f"Agent {agent} episode states: {episode_states[agent]}")
            if len(recent_rewards[agent]) < MAX_NUM_REWARDS_TO_CHECK:
                recent_rewards[agent].append(np.sum(episode_rewards[agent]))
                # print(f"Recent rewards for agent {agent}: {np.sum(episode_rewards[agent])}")
            else:
                recent_rewards[agent].pop(0)
                recent_rewards[agent].append(np.sum(episode_rewards[agent]))
                # print(f"Recent rewards for agent {agent}: {np.sum(episode_rewards[agent])}")

            avg_reward = np.mean(recent_rewards[agent])
            std_reward = np.std(recent_rewards[agent])
            

            if skip_update[agent]:
                print(f"Checking if policy re-training is needed for agent {agent}")
                if avg_reward < rl_params.reward_avg_threshold and std_reward > rl_params.reward_std_threshold:
                    print(f"Policy re-training needed for agent {agent}")
                    skip_update[agent] = False

            if not skip_update[agent]:
                print(f"Updating policy for agent {agent}")
                if avg_reward >= rl_params.reward_avg_threshold and std_reward < rl_params.reward_std_threshold:
                    print(f"Training completed for {agent}")
                    print(f"Average Reward: {avg_reward}, Std Dev Reward: {std_reward}")
                    skip_update[agent] = True

        workload_generator.set_terminated(True)
        time.sleep(15)

        print(f"------------ Iteration {iteration} completed, Results ------------")

        
        all_rewards = initialize_dicts()
        overall_average_rewards = np.mean([reward for episode_rewards in rewards.values() for reward in episode_rewards])
        overall_cumulative_rewards += overall_average_rewards
        writer.add_scalar('Overall/Average_Reward', overall_average_rewards, iteration)
        writer.add_scalar('Overall/Average_Cumulative_Reward', overall_cumulative_rewards, iteration)        
        writer.add_scalar('Overall/Learning_Rate', new_learning_rate, iteration)
        wandb.log({'Overall/Average_Reward': overall_average_rewards, 'iteration': iteration})
        wandb.log({'Overall/Average_Cumulative_Reward': overall_cumulative_rewards, 'iteration': iteration})
        wandb.log({'Overall/Learning_Rate': new_learning_rate, 'iteration': iteration})

        agents_results = {agent: {} for agent in range(num_agents)}

        for agent in range(num_agents):
            average_rewards = np.mean([reward for reward in rewards[agent]])
            iteration_rewards[agent].append(average_rewards)
            smoothed_rewards[agent].append(np.mean(iteration_rewards[agent][-10:]))
            cumulative_rewards[agent] += average_rewards
            
            writer.add_scalar(f'Agent_{agent}/Cumulative_Reward', cumulative_rewards[agent], iteration)
            writer.add_scalar(f'Agent_{agent}/Average_Reward', average_rewards, iteration)
            wandb.log({f'Agent_{agent}/Cumulative_Reward': cumulative_rewards[agent], 'iteration': iteration})
            wandb.log({f'Agent_{agent}/Average_Reward': average_rewards, 'iteration': iteration})
            
            
            print(f' --> Average rewards for agent {agent}: {np.round(average_rewards, decimals=3)}, Moving average: {np.round(np.mean(iteration_rewards[agent][-10:]), decimals=3)}')

            if SAVE_TO_FILE:
                all_rewards[agent] = [reward for reward_ep in rewards[agent] for reward in reward_ep]
                agents[agent].save_trajectories(iteration=iteration, states=states[agent], actions=actions[agent], rewards=all_rewards[agent], log_probs=log_probs[agent])
                print(f'Trajectory for {agent} saved to file')
                
            if skip_update[agent]:
                print(f"Skipping policy update for agent {agent}")
                continue
            
            print(f"Updating policy for agent {agent}")
            result = agents[agent].learn(states[agent], actions[agent], rewards[agent], log_probs[agent], iteration)
            wandb.log({f"Agent_{agent}/Actor Loss": result['actor_loss'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Critic Loss": result['critic_loss'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Total Loss": result['total_loss'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Entropy": result['entropy'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Value Loss": result['value_loss'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Policy Gradient Loss": result['policy_gradient_loss'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Approx KL": result['approx_kl'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Clip Fraction": result['clip_fraction'], "iteration": iteration})
            wandb.log({f"Agent_{agent}/Explained Variance": result['explained_variance'], "iteration": iteration})
        
    
            # Log action distribution
            # for action, count in enumerate(result['action_distribution']):
            #     wandb.log({f"Agent_{agent}_Actions/Action_Distribution": wandb.Histogram(count), "action": action})

            agents_results[agent] = result

            # # Log histograms of network parameters
            # for name, param in agents[agent].actor.named_parameters():
            #     wandb.log({f'Agent_{agent}_Actor/{name}': wandb.Histogram(param.detach().cpu().numpy()), "iteration": iteration})
            # for name, param in agents[agent].critic.named_parameters():
            #     wandb.log({f'Agent_{agent}_Critic/{name}': wandb.Histogram(param.detach().cpu().numpy()), "iteration": iteration})

        # average of measures in Overall
        wandb.log({'Overall/Average_Actor_Loss': np.mean([result['actor_loss'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Critic_Loss': np.mean([result['critic_loss'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Total_Loss': np.mean([result['total_loss'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Entropy': np.mean([result['entropy'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Value_Loss': np.mean([result['value_loss'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Policy_Gradient_Loss': np.mean([result['policy_gradient_loss'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Approx_KL': np.mean([result['approx_kl'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Clip_Fraction': np.mean([result['clip_fraction'] for result in agents_results.values()]), 'iteration': iteration})
        wandb.log({'Overall/Average_Explained_Variance': np.mean([result['explained_variance'] for result in agents_results.values()]), 'iteration': iteration})
        
        # # Log action distribution
        # for action, count in enumerate(np.mean([result['action_distribution'] for result in agents_results.values()], axis=0)):
        #     wandb.log({"Overall_Actions/Action_Distribution": wandb.Histogram(count), "action": str(action + 1)})
        

        if iteration % 5 == 0 and iteration != 0:
            for agent in range(num_agents):
                agents[agent].save_checkpoint(iteration, timestamp_creation=timestamp, model=options.agent)
    
        if iteration % 10 == 0 and iteration != 0:
            undeploy_app(file_deployment, app_namespace)
            time.sleep(10)
            deploy_app(file_deployment, app_namespace)
            env.check_app_ready()
    
    for agent in range(num_agents):     
        if PLOT_FIG:
            visualization(iteration_rewards[agent], smoothed_rewards[agent], rl_params.app_name, agent, rl_params.model, timestamp, results_dir)

        
        if SAVE_TO_FILE:
            save_results(iteration_rewards[agent], smoothed_rewards[agent], app_name, agent, rl_params.model, timestamp, results_dir)
            
            
        #Tensorboard visualize ACTOR and CRITIC
        agents[agent].visualize_tensorboard()
        

    env.close()
            
            
            
def get_addresses(nodes, options, rl_params):
    app_name = rl_params.app_name
    prom_address = None
    call_host = None

    get_node_infos(rl_params, nodes)
    #number of nodes
    rl_params.num_nodes = len(nodes.items)

    if nodes.items:
        first_node = nodes.items[0]
        for address in first_node.status.addresses:
            if address.type == "InternalIP":
                break
        if options.prom_address is not None:
            prom_address = options.prom_address
        else:
            prom_address = f"http://{first_node.status.addresses[0].address}:30090"

        if app_name == "teastore":
            call_host = "http://localhost:30080/tools.descartes.teastore.webui/"
    else:
        print("No nodes found in the cluster")

    return prom_address, call_host

def get_node_infos(rl_params, nodes):
    rl_params.num_nodes = len(nodes.items)
    rl_params.num_cores = 0
    rl_params.memory = 0
    rl_params.hostnames = []

    for node in nodes.items:
        # Extract hostname
        rl_params.hostnames.append(node.metadata.name)
        
        # Extract CPU and Memory information
        cpu_capacity = node.status.capacity.get("cpu")
        mem_capacity = node.status.capacity.get("memory")

        if cpu_capacity:
            rl_params.num_cores += int(cpu_capacity)
        
        if mem_capacity:
            # Memory is usually given in Ki, we convert it to Gi
            rl_params.memory += int(mem_capacity[:-2]) // (1024 * 1024)

        # Extract GPU information if available
        gpu_capacity = node.status.capacity.get("nvidia.com/gpu")
        if gpu_capacity:
            rl_params.gpu_name = "nvidia.com/gpu"
            rl_params.num_gpus += int(gpu_capacity)



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

    # Arguments related to the application
    parser.add_argument('--app_name', type=str, default='teastore', help='Name of the app to control')
    parser.add_argument('--file_deployment', type=str, default="teastore-1c-5gib.yaml", help='File for deployment, in yaml format')
    parser.add_argument('--app_namespace', type=str, default='default', help='Namespace of the app')
    parser.add_argument('--use_inference', action='store_true', help='True for skipping RL training, default False')
    parser.add_argument('--use_checkpoint', action='store_true', help='True for loading from a model checkpoint, default False')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--group_file', type=str, default="./groups.txt", help='Number of groups in the app')
    parser.add_argument('--prom_address', type=str, help='Prometheus address')
    parser.add_argument('--agent', type=str, default='ppo_aware', help='Name of the agent to use')
    parser.add_argument('--save_to_file', type=lambda x: bool(strtobool(x)), default=True, help='Save results to file')
    parser.add_argument('--plot_fig', type=lambda x: bool(strtobool(x)), default=True, help='Plot figures')

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--k_epochs', type=int, default=5, help='Update policy for K epochs')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clipping ratio for PPO')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max norm for gradient clipping')
    parser.add_argument('--critic_loss_discount', type=float, default=0.05, help='Discount factor for critic loss')
    parser.add_argument('--minibatch_size', type=int, default=5, help='Mini-batch size')
    parser.add_argument('--max_num_rewards_to_check', type=int, default=10, help='Number of rewards to check')
    parser.add_argument('--gae_lam' , type=float, default=0.95, help='Generalized Advantage Estimation lambda')

    # Training parameters
    parser.add_argument('--total_iterations', type=int, default=1000, help='Total iterations')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_steps_per_episode', type=int, default=200, help='Max timesteps in one episode')
    parser.add_argument('--update_timestep', type=int, default=4000, help='Timestep threshold for updating policy')
    parser.add_argument('--save_interval', type=int, default=50, help='Save interval')
    parser.add_argument('--model', type=str, default='ppo_aware', help='Model to use')
    parser.add_argument('--reward_type', type=str, default='aware', help='Type of reward to use, either simple, hard or aware')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_tries_cluster_stable', type=int, default=8, help='Number of tries to check if cluster is stable')
    
    # Policy network parameters
    parser.add_argument('--hidden_size', type=int, default=64, help='Number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')

    # Retraining monitoring thresholds
    parser.add_argument('--reward_avg_threshold', type=int, default=100, help='Average reward threshold for retraining')
    parser.add_argument('--reward_std_threshold', type=int, default=10, help='Standard deviation threshold for retraining')

    # Machine parameters

    
    args = parser.parse_args()
    return args


def visualization(iteration_rewards, smoothed_rewards ,app_name ,agent, model, timestamp, dir):
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
    plt.close()


def save_results(iteration_rewards, smoothed_rewards, app_name ,agent, model, timestamp, dir):
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
    
def undeploy_app(file_deployment, app_namespace):
    #to customize. DEFAULT TEASTORE
    print("Undeploying app")
    os.system(f"kubectl delete -f ./apps/{file_deployment} -n {app_namespace}")
    print("App undeployed")

def save_experiment_parameters(log_dir, app_name,timestamp, rl_params):
    log_dir = os.path.join(log_dir, app_name, str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/experiment_parameters.json', 'w') as f:
        json.dump(rl_params.__dict__, f, indent=2)


if __name__ == '__main__':
    main()
