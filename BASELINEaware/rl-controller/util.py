import numpy as np


"""
State and Action Space
"""
NUM_STATES = 5  # TODO: To be confirmed
NUM_ACTIONS = 7
VERTICAL_SCALING_STEP = 128
HORIZONTAL_SCALING_STEP = 1

"""
Meta-learning
"""
BUFFER_UPDATE_MODE = 'best'
BUFFER_SIZE = 32

NUM_FEATURES_PER_SHOT = 13  # 11 (obs) + 1 (action) + 1 (reward)
RNN_HIDDEN_SIZE = 256  # equal to the max steps of each episode
RNN_NUM_LAYERS = 2
EMBEDDING_DIM = 32

MAX_TIMESTEPS_PER_EPISODE = 10  # equal to EPISODE_LENGTH
MAX_NUM_TIMESTEPS = 100000000

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
Hyperparameters
"""
TOTAL_ITERATIONS = 1
EPISODES_PER_ITERATION = 5
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


# return the current states (dictionary) in the form of a vector
def convert_state_dict_to_list(state):
    # TODO: normalize each variable in the state vector
    return list(state.values())

# convert the state (dictionary) to a string
def state_to_string(state):
    state_string = 'State: \n' +\
            'Avg CPU utilization: {:.3f}\n'.format(state['cpu_util']) +\
            'Avg memory utilization: {:.3f}\n'.format(state['memory_util']) +\
            'Num of replicas: {:d}\n'.format(state['num_replicas']) +\
            'CPU limit: {:d}\n'.format(state['cpu_limit']) +\
            'Memory limit: {:d}'.format(state['memory_limit'])
    return state_string


 #  'Avg disk I/O usage: {:.3f}\n'.format(state['disk_io_usage']) +\
            #  'Avg file discovery rate: {:.3f}\n'.format(state['file_discovery_rate']) +\
            #  'Avg rate: {:.3f}\n'.format(state['rate']) +\
            #  'Avg processing rate: {:.3f}\n'.format(state['processing_rate']) +\
            #  'Avg ingestion rate: {:.3f}\n'.format(state['ingestion_rate']) +\
            #  'Avg latency: {:.3f}\n'.format(state['rate']) +\

# print (state, action, reward) for the current step
def print_step_info(step, state_dict, action_dict, reward):
    state = state_to_string(state_dict)
    action = 'Action: N/A'
    if action_dict['vertical_cpu'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' CPU limit'
    elif action_dict['vertical_cpu'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' CPU limit'
    elif action_dict['vertical_memory'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' memory limit'
    elif action_dict['vertical_memory'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' memory limit'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' replicas'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' replicas'
    print('Step #' + str(step))
    print(state)
    print(action, '| Reward:', reward)

# define reward functions
# CHANGEME: [Define customized SLO-driven reward function here]
# calculate the reward based on the current state (after the execution of the current action)
# + utilization [0, 1]
# + data processing rate [0, 1]
# - penalty
# v1: R = alpha * RU + (1-alpha) * DP - penalty
def convert_state_action_to_reward(state, action, last_action, last_state, app_name='my-app'):
    alpha = 0.3
    print(state['cpu_util'] , state['memory_util'])

    resource_util_score = state['cpu_util'] + state['memory_util'] / 2.0
    print(resource_util_score)
    # if state['ingestion_rate'] == 0:
    #     data_processing_rate = 1
    # else:
    #     data_processing_rate = state['processing_rate'] / state['ingestion_rate']

    # reward function definition
    #should be data_processing_rate instead of 1

    print(resource_util_score)

    reward = alpha * resource_util_score + (1-alpha) * 1

    print(reward)
    #Print the whole computation with the numbers instead of the variables for the reward

    # give penalty to frequent dangling actions: e.g., scale in and out
    if last_action['horizontal'] * action['horizontal'] < 0:
        print("Horizontal penalty")
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_cpu'] * action['vertical_cpu'] < 0:
        print("Vertical CPU penalty")
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_memory'] * action['vertical_memory'] < 0:
        print("Vertical Memory penalty")
        reward += -ILLEGAL_PENALTY

   
    #lag_increased = True if state['rate'] < last_state['rate'] else False

    # give penalty to any lag increase
    # if lag_increased:
    #     reward += -ILLEGAL_PENALTY

    # if latency-SLO is defined, add SLO preservation ratio as reward as well
    # reward += (1 - alpha) * SLO_Latency / state['latency']

    return reward

# v2: R = RU * DP - penalty
def convert_state_action_to_reward_v2(state, action, last_action, last_state, app_name='my-app'):
    resource_util_score = state['cpu_util'] + state['memory_util'] / 2.0
    if state['ingestion_rate'] == 0:
        data_processing_rate = 1
    else:
        data_processing_rate = state['processing_rate'] / state['ingestion_rate']

    # reward function definition
    reward = resource_util_score * data_processing_rate

    # give penalty to frequent dangling actions: e.g., scale in and out
    if last_action['horizontal'] * action['horizontal'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_cpu'] * action['vertical_cpu'] < 0:
        reward += -ILLEGAL_PENALTY
    elif last_action['vertical_memory'] * action['vertical_memory'] < 0:
        reward += -ILLEGAL_PENALTY

    lag_increased = True if state['rate'] < last_state['rate'] else False

    # give penalty to any lag increase
    if lag_increased:
        reward += -ILLEGAL_PENALTY

    return reward

# count the number of parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)












import time

from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
from prom_crawler_new import PromCrawlerNew
import functools
import random
from copy import copy

import numpy as np
from gymnasium import spaces
from concurrent.futures import ThreadPoolExecutor
from threading import Timer
import random

from kubernetes import client, config

from utils import *


# the class is indeed a singleton. In fact, the environment has to be shared between agents and it is initialized by
# different daemons. The first that actually initializes it wins. The other ones will receive the handlers


class ClusterEnvironment(ParallelEnv, metaclass=SingletonMeta):
    app_name = "app_name"
    app_namespace = "app_namespace"

    initial_num_replicas = 1
    # maybe other states like cpu mem ,,,

    controlled_resources = ['cpu', 'memory'
                            # 'fs',
                            # 'ingress','egress'
                            ]

    custom_metrics = [
        # latency, throughput, dependencies_that_could_help_us
    ]


    last_action = {
        'num_replicas': 1,
        # 'cpu_limit': 1024,
        # 'memory_limit': 1024
    }
    last_reward = 0

    """The metadata holds environment constants.

        The "name" metadata allows the environment to be pretty printed.
        """

    metadata = {'name': 'cluster_environment'}

    def __init__(self, groups, prom_address, kube_client, kube_config, app_name='app_name', app_namespace='app_namespace', limits=None, mock_observation=False):
        """The init method takes in environment arguments.

                Should define the following attributes:
                - num_agents: an integer specifying the number of agents
                - kube_config: a kubernetes configuration object
                - kube_client: a kubernetes client object
                - prom_address: a string specifying the address of the prometheus server (in cluster or outside, it should reflect the microservices status)
                - limits: a dictionary specifying the limits of the environment (e.g. max_replicas, min_replicas, etc.) in the following format:
                    0: {
                        'max_replicas': 10,
                        'min_replicas': 1,
                        ...
                        }
                    1: {
                        'max_replicas': 10,
                        'min_replicas': 1,
                        ...
                        }
                    ...
                    if not provided, the environment will use default values (max_replicas=10, min_replicas=1)
                - TO BE CONTINUED...



                Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
                Spaces should be defined in the action_space() and observation_space() methods.
                If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

                These attributes should not be changed after initialization.
                """

        super().__init__()

        #app variables
        self.app_name = app_name
        self.app_namespace = app_namespace

        #kubeconfig and kube variables
        # config.load_kube_config()

        # self.kube_client = client.ApiClient()
        self.coreV1 = kube_client.CoreV1Api()
        self.appsV1 = kube_client.AppsV1Api()
        self.control_node_name = None
        self.control_node_IP = None
        #check if kube is working
        try:
            nodes = self.coreV1.list_node()
            resp = self.coreV1.list_namespaced_pod(namespace=self.app_namespace)

            #TAKE NODE ADDRESS AND PORT (SHOULD BE 9100)
            if nodes.items:
                first_node = nodes.items[0]
                for address in first_node.status.addresses:
                    if address.type == "InternalIP":
                        # print(f"Node Name: {first_node.metadata.name}, IP Address: {address.address}")
                        self.control_node_name = first_node.metadata.name
                        self.control_node_IP = address.address
                        break

                if prom_address != None:
                    self.prom_address = prom_address
                else:
                    self.prom_address = f"http://{first_node.status.addresses[0].address}:30090"
        except Exception as e:
            print(e)
            raise Exception("Kubernetes is not working")

        
        self.terminated=False
    

        if self.prom_address:
            self.prom_client = PromCrawlerNew(prom_address)
        else:
            self.prom_client = None
            raise Exception("Prometheus address is not provided")
        
        #context variables
        self.workload_on = False



        # RL Variables
        self.groups = groups
        self.limits = limits
        self.mock_observation = mock_observation
        self.possible_agents = [int(i) for i in range(len(self.groups.keys()))]
        self.active_agents = copy(self.possible_agents)
        self.timestep = 0

        print("List of groups: ", self.groups)
        print("List of limits: ", self.limits)

        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.last_metrics_fetched = {}
        print("Environment initialized")
        print("Observation spaces: ", self.observation_spaces)
        print("Action spaces: ", self.action_spaces)

        # action and observation spaces defined as functions

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # we could add CPU and memory later in other versions, we should use Multi discrete function. For now Just
        # Discrete

        if self.limits:
            max_replicas = self.limits[agent]['max_replicas']
            min_replicas = self.limits[agent]['min_replicas']
            # add here other limits
        else:
            max_replicas = MAX_REPLICAS
            min_replicas = MIN_REPLICAS
            # add here other limits

        return spaces.Discrete(max_replicas, start=min_replicas)

    @functools.lru_cache(maxsize=None)
    def action_mask(self, agent):
        pass
        # This function should return the action masking of the environment
        # potentially we could use this function to mask the actions that are not allowed in the environment, but obviously we mask just actions that are already included in the range of possible actions

    def observation_space(self, agent):
        # This function should return the observation space of the environment
        # The observation space depends on the metrics defined in the environment and other things so for now implemented in mock_observation mode
        if self.mock_observation:
            # for the mock i define an observation space made from 4 metrics
            # avg cpu utilization, avg memory utilization, number of replicas, average_latency
            return spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, float(self.limits[agent]["min_replicas"]), 0.0]),
                              high=np.array([1.0, 1.0,1.0, 1.0, float(self.limits[agent]["max_replicas"]), np.inf], dtype=np.float32))
        else:
            # TODO: Implement the observation space with the real metrics that we want to define. Follow the same pattern as the mock observation space, so low values and high values
            raise Exception("Not implemented yet")

    # should return the observation space of the environment
    # so that the agent can start anew with the environment
    def reset(self, seed=None, options=None):
        # we could put some waiting there or is it given to the fetch metrics?
        self.active_agents = copy(self.possible_agents)  # reset the active agents
        observations = {}
        self.timestep=0
        self.terminated=False

        # optional: wait for the environment to be ready
        self.initialize_k8s_environment()
        print("Every replica set to 1")

        results = self.fetch_metrics()

        for agent in self.active_agents:
            observations[agent] = self.generate_observations(agent, results)


        # time.sleep(10)

        # print("Observations at reset: ", observations)
        return observations



    def step(self, actions):
        print("Step ", self.timestep)
        #check if all replicas are ready for all groups

        with ThreadPoolExecutor() as executor:
            executor.map(self.execute_action_scale_group, actions.keys(), [actions[agent] for agent in actions.keys()])

        with ThreadPoolExecutor() as executor:
            executor.map(self.wait_replicas_group_ready, self.groups.keys(), [actions[agent] for agent in actions.keys()])

        results= self.fetch_metrics()

        observations = {}
        rewards = {}
        terminateds = {}
        infos = {}
        
        for agent in self.active_agents:
            observations[agent] = self.generate_observations(agent, results)
            rewards[agent] = self.compute_reward(agent=agent, observations=observations[agent])
            terminateds[agent] = self.compute_termination(agent=agent)
            infos[agent] = self.compute_infos(observations[agent], rewards[agent], terminateds[agent], agent)        
        
        self.timestep += 1
        self.active_agents = [agent for agent in actions.keys()] #reset the active agents??

        return observations, rewards, terminateds, infos

        # fetch metrics concurrently for all agents


    def initialize_k8s_environment(self):
        # reset the number of replicas of the microservices to 1
        with ThreadPoolExecutor() as executor:
            for group in self.groups.keys():
                for microservice in self.groups[group]:
                    executor.submit(self.change_replicas_microservice, microservice, 1, wait=True)

        # print("Every replica reset to 1")

    def change_replicas_microservice(self, microservice, num_replicas=1, wait=True):
        # Reset the number of replicas of the microservice to the initial value
        present = False
        for group in self.groups.keys():
            if microservice in self.groups[group]:
                present = True
                break

        if not present:
            raise Exception("Microservice not found in the groups")

        # scale
        try:
            self.appsV1.patch_namespaced_deployment_scale(name=microservice, namespace=self.app_namespace,
                                                          body={"spec": {"replicas": num_replicas}})
        except Exception as e:
            print(e)
            return

        if wait:
            self.wait_replicas_microservices_ready(microservice, num_replicas)


    def wait_replicas_microservices_ready(self, microservice, num_replicas=1):
        while True:
                time.sleep(TIME_API_CHECK_REPLICAS)
                try:
                    resp = self.appsV1.read_namespaced_deployment(name=microservice, namespace=self.app_namespace)
                    pod_list = self.coreV1.list_namespaced_pod(namespace=self.app_namespace,
                                                               label_selector=f"run={microservice}")
                except Exception as e:
                    print(e)
                    continue

                ready_replicas = resp.status.ready_replicas
                ready_pods = [pod for pod in pod_list.items if pod.status.phase == 'Running' and
                              any(cond.type == 'Ready' and cond.status == 'True' for cond in pod.status.conditions)]
                terminating_pods = [pod for pod in pod_list.items if pod.metadata.deletion_timestamp is not None]

                if ready_replicas == num_replicas and len(ready_pods) == num_replicas and len(terminating_pods) == 0:
                    break

    def execute_action_scale_group(self, agent, action):
        # execute the action for the agent
        with ThreadPoolExecutor() as executor:
            executor.map(self.change_replicas_microservice, self.groups[agent], [action] * len(self.groups[agent]))

    def wait_replicas_group_ready(self, group, num_replicas=1):
        with ThreadPoolExecutor() as executor:
            executor.map(self.wait_replicas_microservices_ready, self.groups[group], [num_replicas] * len(self.groups[group]))

    def generate_observations(self, agent, metrics=None):
        microservices_for_agent = self.groups[agent]
        if self.mock_observation:
            # Mock observation
            return np.array(
                [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.randint(0,1), random.randint(0,1), random.randint(self.limits[agent]["min_replicas"], self.limits[agent]["max_replicas"]),
                 random.uniform(0.0, MAX_MOCK_LATENCY)])
        else:
            #in metrics get all the metrics for the microservices of the agent
            #then return the observation

            selected_avg_cpu = {microservice: metrics['avg_cpu_util_microservice'][microservice] for microservice in microservices_for_agent}
            selected_avg_mem = {microservice: metrics['avg_mem_util_microservice'][microservice] for microservice in microservices_for_agent}
            selected_stability_cpu = {microservice: metrics['stability_index_cpu_microservice'][microservice] for microservice in microservices_for_agent}
            selected_stability_mem = {microservice: metrics['stability_index_mem_microservice'][microservice] for microservice in microservices_for_agent}
            #maybe latency....
            num_replicas = {microservice: metrics['num_replicas'][microservice] for microservice in microservices_for_agent}

            #average by microservice
            avg_cpu = np.mean(list(selected_avg_cpu.values()))
            avg_mem = np.mean(list(selected_avg_mem.values()))
            avg_stability_cpu = np.mean(list(selected_stability_cpu.values()))
            avg_stability_mem = np.mean(list(selected_stability_mem.values()))
            num_replicas_avg = np.mean(list(num_replicas.values()))
            latency_avg = 0.0 #for now

            return np.array([avg_cpu, avg_mem, avg_stability_cpu, avg_stability_mem, num_replicas_avg, 0.0])
            
            #for now we base our observation on the average cpu and memory utilization and the stability index of the microservices of the agent
            
        
    def fetch_metrics(self):

        """
        Fetch the metrics for the microservices in the environment
        
        Returns:
        avg_cpu_util_microservice: a dictionary containing the average CPU utilization for each microservice
        avg_mem_util_microservice: a dictionary containing the average memory utilization for each microservice
        stability_index_cpu_microservice: a dictionary containing the stability index for the CPU utilization for each microservice
        stability_index_mem_microservice: a dictionary containing the stability index for the memory utilization for each microservice
        ALL IN A DICT
        """

        if self.prom_client is None:
            print("Prometheus client not initialized")
            return

        # put metrics in the last_metrics_fetched dictionary
        container_filter = f"{self.app_name}-.*"
        node_filter = f"{self.control_node_name}"

        avg_cpu_util_microservice = {}
        avg_mem_util_microservice = {}
        stability_index_cpu_microservice = {}
        stability_index_mem_microservice = {}
        num_replicas = {}

        metrics = {
                'metricCPU': metric_cpu.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW, cpu_rate_window=CPU_RATE_WINDOW),
                'metricMEM': metric_mem.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW),
                'metricSTDCPU': metric_stddev_cpu.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_CPU_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW, cpu_rate_window=CPU_RATE_WINDOW),
                'metricSTDMEM': metric_stddev_mem.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_MEM_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW),
                # latency metric TBD
                # 'metricLATENCY': metric_latency.format(app_microservice=app_name, namespace=namespace)
            }

        for num_tries in range(1, NUM_TRIES_CLUSTER_STABLE + 1):
            fetched_metrics = self.prom_client.fetch_metrics(metrics, step= 3)
            if all(value is None for value in fetched_metrics.values()):
                continue

            for fetched_metric, results in fetched_metrics.items():
                for result in results:
                    microservice = result["metric"]["container"]

                    if microservice not in avg_cpu_util_microservice:
                        avg_cpu_util_microservice[microservice] = []
                        avg_mem_util_microservice[microservice] = []
                        stability_index_cpu_microservice[microservice] = []
                        stability_index_mem_microservice[microservice] = []

                
                    if fetched_metric == "metricCPU":
                        avg_cpu_util_microservice[microservice].append(float(result["values"][0][1]))
                    elif fetched_metric == "metricMEM":
                        avg_mem_util_microservice[microservice].append(float(result["values"][0][1]))
                    elif fetched_metric == "metricSTDCPU":
                        stability_index_cpu_microservice[microservice].append(float(result["values"][0][1]))
                    elif fetched_metric == "metricSTDMEM":
                        stability_index_mem_microservice[microservice].append(float(result["values"][0][1]))
            
            time.sleep(PROMETHEUS_STEP)


        # Average the metrics over the tries, just CPU, MEM for now
        avg_cpu_util_microservice = {microservice: np.mean(values) for microservice, values in avg_cpu_util_microservice.items()}
        avg_mem_util_microservice = {microservice: np.mean(values) for microservice, values in avg_mem_util_microservice.items()}
        stability_index_cpu_microservice = {microservice: np.mean(values) for microservice, values in stability_index_cpu_microservice.items()}
        stability_index_mem_microservice = {microservice: np.mean(values) for microservice, values in stability_index_mem_microservice.items()}

        all_microservices = [microservice for group in self.groups.keys() for microservice in self.groups[group]]

        
        for microservice in all_microservices :
            try:
                resp = self.appsV1.read_namespaced_deployment(name=microservice, namespace=self.app_namespace)
                num_replicas[microservice] = resp.spec.replicas
            except Exception as e:
                print(e)
                num_replicas[microservice] = None

            

        dict = {
            "avg_cpu_util_microservice": avg_cpu_util_microservice,
            "avg_mem_util_microservice": avg_mem_util_microservice,
            "stability_index_cpu_microservice": stability_index_cpu_microservice,
            "stability_index_mem_microservice": stability_index_mem_microservice,
            "num_replicas": num_replicas
        }

        return dict



        

    def compute_reward(self, agent, observations):

        alpha = 0.5
        # compute the reward for the agent
        avg_cpu = observations[0] #between 0 and 100
        avg_mem = observations[1] #between 0 and 100
        stability_cpu = observations[2] #between 0 and 100
        stability_mem = observations[3] #
        num_replicas = observations[4]
        #latency = observations[5]
        reward = 0

        penalty = 0 #expand here...
        
        resource_util_score = CPU_IMPORTANCE * avg_cpu + MEM_IMPORTANCE * avg_mem / 2.0
        reward = alpha * resource_util_score + (1-alpha) * (1 - stability_cpu) * (1 - stability_mem) - penalty

        #really we can go deeper here


        
        return reward
    


    def compute_termination(self,agent):
        # compute termination for the environment
        return self.terminated
    
    def set_termination (self):
        #executes a timer that sleeps for a certain amount of time and then sets the termination to True
        self.terminated = True

    def clock_termination(self):
        #executes a timer that sleeps for a certain amount of time and then sets the termination to True
        timer = Timer(TIME_TERMINATED, self.set_termination).start()
        


    def compute_infos(observations, reward, terminated, agent):
        pass