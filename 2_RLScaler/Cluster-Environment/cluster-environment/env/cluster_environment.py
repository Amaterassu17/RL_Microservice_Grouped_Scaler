from utils import *
import time
import functools
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Timer
import numpy as np
from gymnasium import spaces
from kubernetes import client, config
from copy import copy
from pettingzoo import ParallelEnv
from prom_crawler_new import PromCrawlerNew
from utils import *
import math
from math import ceil
import scipy.stats as stats
import wandb



# the class is indeed a singleton. In fact, the environment has to be shared between agents and it is initialized by
# different daemons. The first that actually initializes it wins. The other ones will receive the handlers


class ClusterEnvironment(ParallelEnv, metaclass=SingletonMeta):
    app_name = "app_name"
    app_namespace = "app_namespace"
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

    def __init__(self, rl_params ,groups, prom_address, kube_client, limits=None):
        """The init method takes in environment arguments.

                Should define the following attributes:
                - rl_parans: dict of rl_parameters
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
                - reward_type: a string specifying the type of reward to be used in the environment (either simple, hard, aware)
                - TO BE CONTINUED...



                Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
                Spaces should be defined in the action_space() and observation_space() methods.
                If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

                These attributes should not be changed after initialization.
                """

        super().__init__()
    #app variables
        self.rl_params = rl_params
        self.app_name = self.rl_params.app_name
        self.app_namespace = self.rl_params.app_namespace
    #kubeconfig and kube variables
    # config.load_kube_config()
        self.coreV1 = kube_client.CoreV1Api()
        self.appsV1 = kube_client.AppsV1Api()
        self.control_node_name = None
        self.control_node_IP = None
    #check if kube is working
        try:
            nodes = self.coreV1.list_node()
            resp = self.coreV1.list_namespaced_pod(namespace=self.app_namespace)
            if nodes.items:
                first_node = nodes.items[0]
                for address in first_node.status.addresses:
                    if address.type == "InternalIP":
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
    
        if self.prom_address:
            self.prom_client = PromCrawlerNew(prom_address)
        else:
            self.prom_client = None
            raise Exception("Prometheus address is not provided")
        
        #context variables
        self.workload_on = False
        self.terminated=False
        self.truncated=False

        # RL Variables
        self.groups = groups
        self.limits = limits
        self.possible_agents = [int(i) for i in range(len(self.groups.keys()))]
        self.agents = copy(self.possible_agents)
        # self.active_agents = copy(self.agents)
        self.timestep = 0

        self.reward_type = self.rl_params.reward_type
        self.num_tries_cluster_stable = self.rl_params.num_tries_cluster_stable
        print("List of groups: ", self.groups)
        print("List of limits: ", self.limits)

        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        print("Action spaces: ", self.action_spaces)
        print("Observation spaces: ", self.observation_spaces)
        self.last_state = None
        print("Environment initialized")
        # print("Observation spaces: ", self.observation_spaces)
        # print("Action spaces: ", self.action_spaces)

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # This function should return the observation space of the environment
        # The observation space depends on the metrics defined in the environment and other things so for now implemented in mock_observation mode
    
            # for the mock i define an observation space made from 4 metrics
            # avg cpu utilization, avg memory utilization, number of replicas, average_latency
        # return spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, int(self.limits[agent]["min_replicas"]), 
        #                                 # 0.0
        #                                 ]),
        #                     high=np.array([100.0, 100.0,100.0, 100.0, int(self.limits[agent]["max_replicas"]), 
        #                                 #    MAX_MOCK_LATENCY
        #                                    ], dtype=np.float64))
        avges = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([2.0, 2.0]), dtype=np.float64)
        stabilities = spaces.Box(low=np.array([0,0]), high=np.array([2.0,2.0]), dtype=np.float64)
        num_replicas = spaces.Discrete(int(self.limits[agent]["max_replicas"]), start=int(self.limits[agent]["min_replicas"]))
        latency = spaces.Box(low=np.array([0.0,
                                          0.0,
                                        #    0.0
                                           ]), high=np.array([1.0,
                                                           1.0,
                                                            #   MAX_MOCK_LATENCY
                                                              ]), dtype=np.float64)
        scaling_time = spaces.Box(low=np.array([0.0]), high=np.array([100.0]), dtype=np.float64)
        return spaces.Tuple((avges,  
                             num_replicas, 
                             latency,
                            #  scaling_time
                             ))

    


#IMPORTANT FUNCTIONS, RESET AND STEP

    # should return the observation space of the environment
    # so that the agent can start anew with the environment
    def reset(self, seed=None, options=None):
        # we could put some waiting there or is it given to the fetch metrics?
        self.active_agents = copy(self.possible_agents)  # reset the active agents
        observations = {}
        infos = {}
        self.timestep=0
        self.terminated=False
        self.truncated=False

        # optional: wait for the environment to be ready
        
        scaling_times_groups = {group: 0 for group in self.groups.keys()}

        self.initialize_k8s_environment(scaling_times_groups)
        print("Every replica set to 1")
        results = self.fetch_metrics()
        results['scaling_times_groups'] = scaling_times_groups

        for agent in self.active_agents:
            observations[agent] = self.generate_observations(agent, results)
            infos[agent] = self.compute_infos(observations[agent], 0, False, agent)

        # print("Observations at reset: ", observations)

        self.last_state = observations



        # time.sleep(10)

        # print("Observations at reset: ", observations)
        return observations, infos



    def step(self, actions):
        #check if all replicas are ready for all groups

        scaling_times_groups = {group: 0 for group in self.groups.keys()}

        with ThreadPoolExecutor() as executor:
            futures = []
            start_times = {group: time.time() for group in self.groups.keys()}
            for group, agent in zip(self.groups.keys(), actions.keys()):
                future = executor.submit(self.execute_action_scale_group, group, actions[agent], scaling_times_groups, start_times)
                futures.append(future)
            for future in futures:
                future.result()  # Wait for all actions to complete
                

        # with ThreadPoolExecutor() as executor:
        #     executor.map(self.wait_replicas_group_ready, self.groups.keys(), [actions[agent] for agent in actions.keys()])

        results= self.fetch_metrics()
        results['scaling_times_groups'] = scaling_times_groups

        observations = {}
        rewards = {}
        additional_infos = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        for agent in self.active_agents:
            observations[agent] = self.generate_observations(agent, results)
            rewards[agent], additional_infos[agent] = self.compute_reward(agent=agent, observations=observations[agent], prev_observations = self.last_state[agent], type=self.reward_type)
            terminateds[agent] = self.compute_termination()
            truncateds[agent] = self.compute_truncation()
            infos[agent] = self.compute_infos(observations[agent], rewards[agent], terminateds[agent], agent)    

        # print(f"--- Observations at step {self.timestep}: {observations} ---")
        # print(f"--- Rewards at step {self.timestep}: {rewards} ---")

        
        self.timestep += 1
        self.active_agents = [agent for agent in actions.keys()] #reset the active agents??

        self.last_state = observations

        return observations, rewards, terminateds,truncateds , additional_infos

        # fetch metrics concurrently for all agents


#UTILITY FUNCTIONS

    def initialize_k8s_environment(self, groups_times):
        with ThreadPoolExecutor() as executor:
            futures = []
            start_times = {group: time.time() for group in self.groups.keys()}
            for group in self.groups.keys():
                for microservice in self.groups[group]:
                    future = executor.submit(self.change_replicas_microservice, microservice, 1, wait=True)
                    futures.append((future, group, microservice))

            for future, group, microservice in futures:
                future.result()  # Wait for the operation to complete
                groups_times[group] = time.time() - start_times[group]

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
                                                          body={"spec": {"replicas": int(num_replicas)}})
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

    def execute_action_scale_group(self, group, action, scaling_times_groups, start_times):
        with ThreadPoolExecutor() as executor:
            futures = []
            for microservice in self.groups[group]:
                future = executor.submit(self.change_replicas_microservice, microservice, action, wait=True)
                futures.append((future, microservice))

            for future, microservice in futures:
                future.result()  # Wait for the operation to complete
            scaling_times_groups[group] = time.time() - start_times[group]
        

    def wait_replicas_group_ready(self, group, num_replicas=1):
        with ThreadPoolExecutor() as executor:
            executor.map(self.wait_replicas_microservices_ready, self.groups[group], [num_replicas] * len(self.groups[group]))

    def generate_observations(self, agent, metrics=None):
        microservices_for_agent = self.groups[agent]
        
        
        #in metrics get all the metrics for the microservices of the agent
        #then return the observation

        # selected_avg_cpu = {microservice: metrics['avg_cpu_util_microservice'][microservice] for microservice in microservices_for_agent}
        # selected_avg_mem = {microservice: metrics['avg_mem_util_microservice'][microservice] for microservice in microservices_for_agent}
        # selected_stability_cpu = {microservice: metrics['stability_index_cpu_microservice'][microservice] for microservice in microservices_for_agent}
        # selected_stability_mem = {microservice: metrics['stability_index_mem_microservice'][microservice] for microservice in microservices_for_agent}
        selected_avg_cpu_limit_ratio = {microservice: metrics['cpu_limit_microservice'][microservice] for microservice in microservices_for_agent}
        selected_avg_mem_limit_ratio = {microservice: metrics['mem_limit_microservice'][microservice] for microservice in microservices_for_agent}
        latency50 = metrics['latency_50th_percentile_microservice']
        latency95 = metrics['latency_95th_percentile_microservice']
        latency99 = metrics['latency_99th_percentile_microservice']
        num_replicas = {microservice: metrics['num_replicas'][microservice] for microservice in microservices_for_agent}
        scaling_times_group = metrics['scaling_times_groups'][agent]
        # print(num_replicas)
        #average by microservice
        # avg_cpu = np.mean(list(selected_avg_cpu.values()))
        # avg_mem = np.mean(list(selected_avg_mem.values()))
        # avg_stability_cpu = np.mean(list(selected_stability_cpu.values()))
        # avg_stability_mem = np.mean(list(selected_stability_mem.values()))
       
        try: 
            num_replicas_avg = np.mean(list(num_replicas.values()))
        except Exception as e:
            print(num_replicas)
            print(e)
            exit(1)
       
        avg_cpu_limit_ratio = np.mean(list(selected_avg_cpu_limit_ratio.values()))
        avg_mem_limit_ratio = np.mean(list(selected_avg_mem_limit_ratio.values()))
        
        if avg_cpu_limit_ratio > 2:
            print("CPU limit ratio is greater than 2")
            avg_cpu_limit_ratio = 2
        if avg_mem_limit_ratio > 2:
            print("Memory limit ratio is greater than 2")
            avg_mem_limit_ratio = 2
        
        
        SLO_latency_50 = 20 #ms
        SLO_latency_99 = 5000
        
        if latency50 == 0:
            latency50 = 1
        if latency99 == 0:
            latency99 = 1
        
        latency50_ratio = SLO_latency_50/latency50
        latency99_ratio = SLO_latency_99/latency99
        
    
        if latency50_ratio > 1 or latency50_ratio == np.nan:
            latency50_ratio = 1
        if latency99_ratio > 1 or latency99_ratio == np.nan:
            latency99_ratio = 1
        
    
        return (np.array([avg_cpu_limit_ratio, avg_mem_limit_ratio], dtype=np.float64), int(num_replicas_avg),
        np.array([latency50_ratio,
                  latency99_ratio,
                 #  latency99
                  ], dtype=np.float64),
        # np.array([scaling_times_group], dtype=np.float64)
        )

        


              
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
        cpu_rate_window = 15
        latency_rate_window = 15
        latency_percentiles = [0.50, 0.95, 0.99]


        avg_cpu_util_microservice = {}
        avg_mem_util_microservice = {}
        stability_index_cpu_microservice = {}
        stability_index_mem_microservice = {}
        num_replicas = {}
        cpu_limit_microservice = {}
        mem_limit_microservice = {}
        latencies50 = []
        latencies95 = []
        latencies99 = []
        num_replicas = {}

        metrics = {
                'metricCPU': metric_cpu.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW, cpu_rate_window=cpu_rate_window),
                'metricMEM': metric_mem.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW),
                'metricSTDCPU': metric_stddev_cpu.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_CPU_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW, cpu_rate_window=cpu_rate_window),
                'metricSTDMEM': metric_stddev_mem.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_MEM_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW),
                # latency metric TBD
                'metricLATENCY50': metric_latency.format(app_microservice=container_filter, latency_percentile=latency_percentiles[0], latency_rate_window = latency_rate_window),
                'metricLATENCY95': metric_latency.format(app_microservice=container_filter, latency_percentile=latency_percentiles[1], latency_rate_window = latency_rate_window),
                'metricLATENCY99': metric_latency.format(app_microservice=container_filter, latency_percentile=latency_percentiles[2], latency_rate_window = latency_rate_window),
                'metricCPULimit' : metric_usage_cpu_hpa_formula.format(app_microservice=container_filter, control_node=node_filter, cpu_rate_window=cpu_rate_window),
                'metricMEMLimit' : metric_usage_mem_hpa_formula.format(app_microservice=container_filter, control_node=node_filter),
                'num_replicas': metric_replicas.format(app_microservice=container_filter),
            }        
    
        

        for num_tries in range(1, self.num_tries_cluster_stable + 1):
            fetched_metrics = self.prom_client.fetch_metrics(metrics, step= 3)
            if all(value is None for value in fetched_metrics.values()):
                continue
            

            for fetched_metric, results in fetched_metrics.items():
                for result in results:
                    if result["metric"] != {}:
                        microservice = result["metric"]["container"]
                   
                    

                        if microservice not in avg_cpu_util_microservice:
                            avg_cpu_util_microservice[microservice] = []
                            avg_mem_util_microservice[microservice] = []
                            stability_index_cpu_microservice[microservice] = []
                            stability_index_mem_microservice[microservice] = []
                            cpu_limit_microservice[microservice] = []
                            mem_limit_microservice[microservice] = []
                            num_replicas[microservice] = 1
                        
                        # print(num_replicas[microservice])
                 
                
                        if fetched_metric == "metricCPU":
                            avg_cpu_util_microservice[microservice].append(float(result["values"][0][1]))
                        elif fetched_metric == "metricMEM":
                            avg_mem_util_microservice[microservice].append(float(result["values"][0][1]))
                        elif fetched_metric == "metricSTDCPU":
                            stability_index_cpu_microservice[microservice].append(float(result["values"][0][1]))
                        elif fetched_metric == "metricSTDMEM":
                            stability_index_mem_microservice[microservice].append(float(result["values"][0][1]))
                        # elif fetched_metric == "metricLATENCY":
                        #     latency_microservice[microservice].append(float(result["values"][0][1]))
                        elif fetched_metric == "metricCPULimit":
                            cpu_limit_microservice[microservice].append(float(result["values"][0][1]))
                        elif fetched_metric == "metricMEMLimit":
                            mem_limit_microservice[microservice].append(float(result["values"][0][1]))
                            
                    if fetched_metric == "metricLATENCY50":
                        # print(f"Latency 50th percentile for {microservice}: {result}")
                        if not math.isnan(float(result["values"][0][1])):
                            latencies50.append(float(result["values"][0][1]))
                    elif fetched_metric == "metricLATENCY95":
                        if not math.isnan(float(result["values"][0][1])):
                            latencies95.append(float(result["values"][0][1]))
                    elif fetched_metric == "metricLATENCY99":
                        if not math.isnan(float(result["values"][0][1])):
                            latencies99.append(float(result["values"][0][1]))
                        
                    elif fetched_metric == "num_replicas":
                        # print(f"Num replicas for {microservice}: {result}")
                        num_replicas[microservice] = int(result["values"][0][1])
            
            time.sleep(PROMETHEUS_STEP)

        # filter empty microservices in dictionaries
        avg_cpu_util_microservice = {microservice: values for microservice, values in avg_cpu_util_microservice.items() if values}
        avg_mem_util_microservice = {microservice: values for microservice, values in avg_mem_util_microservice.items() if values}
        stability_index_cpu_microservice = {microservice: values for microservice, values in stability_index_cpu_microservice.items() if values}
        stability_index_mem_microservice = {microservice: values for microservice, values in stability_index_mem_microservice.items() if values}
        cpu_limit_microservice = {microservice: values for microservice, values in cpu_limit_microservice.items() if values}
        mem_limit_microservice = {microservice: values for microservice, values in mem_limit_microservice.items() if values}
        num_replicas = {microservice: num_replicas[microservice] for microservice in avg_cpu_util_microservice.keys()}

        # print if values are None or arrays are empty
        if not avg_cpu_util_microservice:
            print("avg_cpu_util_microservice is empty")
        if not avg_mem_util_microservice:
            print("avg_mem_util_microservice is empty")
        if not stability_index_cpu_microservice:
            print("stability_index_cpu_microservice is empty")
        if not stability_index_mem_microservice:
            print("stability_index_mem_microservice is empty")
        if not cpu_limit_microservice:
            print("cpu_limit_microservice is empty")
        if not mem_limit_microservice:
            print("mem_limit_microservice is empty")
        if not num_replicas:
            print("num_replicas is empty")


        avg_cpu_util_microservice = {microservice: np.mean(values) for microservice, values in avg_cpu_util_microservice.items()}
        avg_mem_util_microservice = {microservice: np.mean(values) for microservice, values in avg_mem_util_microservice.items()}
        stability_index_cpu_microservice = {microservice: np.mean(values) for microservice, values in stability_index_cpu_microservice.items()}
        stability_index_mem_microservice = {microservice: np.mean(values) for microservice, values in stability_index_mem_microservice.items()}
        cpu_limit_microservice = {microservice: np.mean(values) for microservice, values in cpu_limit_microservice.items()}
        mem_limit_microservice = {microservice: np.mean(values) for microservice, values in mem_limit_microservice.items()}
        
        #add print latencies IF they are empty
        
        # add deployed ready replicas to get num_replicas of the application with kube and using try and except
    
        
        for microservice in avg_cpu_util_microservice.keys():
            while True:
                try:
                    resp = self.appsV1.read_namespaced_deployment(name=microservice, namespace=self.app_namespace)
                    num_replicas[microservice] = resp.status.ready_replicas
                    if num_replicas[microservice] is not None:
                        break
                except Exception as e:
                    print(e)

        if latencies50 == []:
            print("latencies50 is empty or has None")
            latency50 = 0
        else:
            latency50 = np.mean(latencies50)
        
        if latencies95 == []:
            print("latencies95 is empty or has None")
            latency95 = 0
        else:
            latency95 = np.mean(latencies95)
            
        if latencies99 == []:
            print("latencies99 is empty or has None")
            latency99 = 0
        else:
            latency99 = np.mean(latencies99)
            
            
        num_replicas = {microservice: num_replicas[microservice] for microservice in avg_cpu_util_microservice.keys()}

        dict = {
            "avg_cpu_util_microservice": avg_cpu_util_microservice,
            "avg_mem_util_microservice": avg_mem_util_microservice,
            "stability_index_cpu_microservice": stability_index_cpu_microservice,
            "stability_index_mem_microservice": stability_index_mem_microservice,
            "num_replicas": num_replicas,
            "cpu_limit_microservice": cpu_limit_microservice,
            "mem_limit_microservice": mem_limit_microservice,
            "latency_50th_percentile_microservice": latency50,
            "latency_95th_percentile_microservice": latency95,
            "latency_99th_percentile_microservice": latency99
        }


        return dict



    
    def compute_reward(self, agent, observations, prev_observations, type="simple2"):

        avg_cpu = observations[0][0] #between 0 and 1, in our case it's the ratio between the cpu usage and the limit of the pod
        avg_mem = observations[0][1] #between 0 and 1, in out case it's the ratio between the cpu usage and the limit of the pod
        num_replicas = observations[1]
        latency50_ratio = observations[2][0] #between 0 and 1, in our case it's the SLO ratio that defines how good is the 50th percentile latency SLO respected
        # latency95 = observations[2][1]
        latency99_ratio = observations[2][1] #between 0 and 1, in our case it's the SLO ratio that defines how good is the 99th percentile in latency SLO respected
        # scaling_time = observations[3][0]

        last_avg_cpu = prev_observations[0][0] #between 0 and 1, in our case it's the ratio between the cpu usage and the limit of the pod
        last_avg_mem = prev_observations[0][1] #between 0 and 1, in out case it's the ratio between the cpu usage and the limit of the pod
        last_num_replicas = prev_observations[1]
        last_latency50_ratio = prev_observations[2][0] #between 0 and 1, in our case it's the SLO ratio that defines how good is the 50th percentile latency SLO respected
        # last_latency95 = prev_observations[2][1]
        last_latency99_ratio = prev_observations[2][1] #between 0 and 1, in our case it's the SLO ratio that defines how good is the 99th percentile in latency SLO respecte
        # old_scaling_time = prev_observations[3][0]
        
        target_cpu = 0.8 #80% of the cpu limit
        target_mem = 0.8 #80% of the memory limit


        current_desired_replica_cpu = ceil(num_replicas * (avg_cpu / target_cpu))
        current_desired_replica_mem = ceil(num_replicas * (avg_mem / target_mem))
        
        old_desired_replica_cpu = ceil(last_num_replicas * (last_avg_cpu / target_cpu))
        old_desired_replica_mem = ceil(last_num_replicas * (last_avg_mem / target_mem))
        
        old_best_replica = max(old_desired_replica_cpu, old_desired_replica_mem)
        
        min_replicas_limit = self.limits[agent]['min_replicas']
        max_replicas_limit = self.limits[agent]['max_replicas']
        
        
        before_cpu_utilization_reward = self.positive_skew_function(last_avg_cpu, target_cpu)
        before_mem_utilization_reward = self.positive_skew_function(last_avg_mem, target_mem)
        
        after_cpu_utilization_reward = self.positive_skew_function(avg_cpu, target_cpu)
        after_mem_utilization_reward = self.positive_skew_function(avg_mem, target_mem)
        
        total_cpu_utilization_reward = (0.5 * before_cpu_utilization_reward + 0.5 * after_cpu_utilization_reward)
        total_mem_utilization_reward = (0.5 * before_mem_utilization_reward + 0.5 * after_mem_utilization_reward) 
        
        type_of_scale = num_replicas - last_num_replicas
        difference_action_desired = abs(num_replicas - old_best_replica)
        
        
        cpu_difference_ratio = (avg_cpu - last_avg_cpu) / last_avg_cpu
        mem_difference_ratio = (avg_mem - last_avg_mem) / last_avg_mem

        cpu_similar_threshold = 0.2
        mem_similar_threshold = 0.2
            
        latency50_difference = latency50_ratio - last_latency50_ratio
        latency99_difference = latency99_ratio - last_latency99_ratio

        reward = 0
        normalized_reward = 0
        additional_info = dict()
        additional_info['old_best_replica'] = old_best_replica
        additional_info['num_replicas'] = num_replicas
        
        
        
        

        if type == "simple":
            #AWARE REWARD

            weight_difference_replica = 0.5

            # Calculate desired replicas based on current observations
            current_desired_replica_cpu = ceil(num_replicas * (avg_cpu / target_cpu))
            current_desired_replica_mem = ceil(num_replicas * (avg_mem / target_mem))

            print(f"--- ---Current desired replica CPU: {num_replicas} * ({avg_cpu} / {target_cpu}) = {current_desired_replica_cpu}")
            print(f"--- ---Current desired replica Memory: {num_replicas} * ({avg_mem} / {target_mem}) = {current_desired_replica_mem}"
                  )
            
            max_current_desired_replica = max(current_desired_replica_cpu, current_desired_replica_mem)

            # Calculate desired replicas based on previous observations
            old_desired_replica_cpu = ceil(last_num_replicas * (last_avg_cpu / target_cpu))
            old_desired_replica_mem = ceil(last_num_replicas * (last_avg_mem / target_mem))

            print(f"--- ---Old desired replica CPU: {last_num_replicas} * ({last_avg_cpu} / {target_cpu}) = {old_desired_replica_cpu}")
            print(f"--- ---Old desired replica Memory: {last_num_replicas} * ({last_avg_mem} / {target_mem}) = {old_desired_replica_mem}")

            # Determine the best replica count from previous desired replicas
            old_best_replica = max(old_desired_replica_cpu, old_desired_replica_mem)
            print(f"--- Old best replica (max of old CPU and Memory replicas): {old_best_replica}")
            
            

            # Penalize large changes in the number of replicas (scaling stability)
            scaling_penalty = abs(num_replicas - old_best_replica)
            print(f"--- --- [NOT USED] Scaling penalty (absolute difference between current and old best replicas): {scaling_penalty}\n")

            cpu_utilization_reward = self.positive_skew_function(avg_cpu, target_cpu)
            mem_utilization_reward = self.positive_skew_function(avg_mem, target_mem)

            print(f"CPU utilization reward: skew{avg_cpu} = {cpu_utilization_reward}")
            print(f"Memory utilization reward: skew{avg_mem} = {mem_utilization_reward}")

            # Reward for correct scaling action
            scaling_action_reward = 0
            difference_action_desired = abs(num_replicas - old_best_replica)
            
            if num_replicas > last_num_replicas:
                print(f"--- ---Scaling up: {num_replicas} > {last_num_replicas}")
                #scale up
                if difference_action_desired >= 0:
                    print(f"--- ---Difference action desired: {difference_action_desired}")
                    scaling_action_reward = np.tanh(-0.5 * difference_action_desired) + 1
                    print(f"--- ---Scaling action reward (scale up): tanh(-0.5 * {difference_action_desired}) + 1 = {scaling_action_reward}")
            elif num_replicas < last_num_replicas:
                #scale down
                print(f"--- ---Scaling down: {num_replicas} < {last_num_replicas}")
                if difference_action_desired <= 0:
                    print(f"--- ---Difference action desired: {difference_action_desired}")
                    scaling_action_reward = np.tanh(-0.5 * difference_action_desired) + 1
                    print(f"--- ---Scaling action reward (scale down): tanh(-0.5 * {difference_action_desired}) + 1 = {scaling_action_reward}")
                    
                    
            elif num_replicas == last_num_replicas:
                scaling_action_reward = 1
            
            #target between 0 and 1
            

            # if last_action > 0 and avg_cpu > target_cpu and avg_mem > target_mem:
            #     # Last action was scale up, and current metrics justify scaling up
            #     scaling_action_reward = 1 * last_action / num_replicas
            #     print(f"Scaling action reward (scale up justified): 1 * {last_action} / {num_replicas} = {scaling_action_reward}")
            # elif last_action < 0 and avg_cpu < target_cpu and avg_mem < target_mem:
            #     # Last action was scale down, and current metrics justify scaling down
            #     scaling_action_reward = 1 * abs(last_action) / num_replicas
            #     print(f"Scaling action reward (scale down justified): 1 * abs({last_action}) / {num_replicas} = {scaling_action_reward}")

            # # Active cluster reward multiplier (higher if the cluster is active and within thresholds)
            # active_cluster_reward_multiplier = 1 + (min(avg_cpu, target_cpu) + min(avg_mem, target_mem)) / (2 * max(target_cpu, target_mem))
            # print(f"Active cluster reward multiplier: 1 + (min({avg_cpu}, {target_cpu}) + min({avg_mem}, {target_mem})) / (2 * max({target_cpu}, {target_mem})) = {active_cluster_reward_multiplier}")


            active_cluster_reward_multiplier = 1

            # Combine the rewards and penalties
            reward = 0.5*(0.5*cpu_utilization_reward + 0.5*mem_utilization_reward) + 0.3*(0.5*(latency50_ratio) + 0.5*(latency99_ratio)) + 0.2*scaling_action_reward
            normalized_reward = np.tanh(reward)
            print(f"Combined reward: 0.7*(0.5*{cpu_utilization_reward} + 0.5*{mem_utilization_reward}) + 0.3*(0.5*({latency50_ratio}) + 0.5*({latency99_ratio})) + 0.2*{scaling_action_reward} = {reward}")
            print(f"norm Reward: {normalized_reward}")
            print(f"""

            --- Agent {agent} --- [{self.groups[agent]}] ---
            Last states:
            Avg CPU: {last_avg_cpu},
            Avg MEM: {last_avg_mem},
            Num replicas: {last_num_replicas}
            ---
            Current states:
            Avg CPU: {avg_cpu},
            Avg MEM: {avg_mem},
            Num replicas: {num_replicas}
            ---
            Reward : {reward}
            --- cpu_utilization_reward: {cpu_utilization_reward},
            --- mem_utilization_reward: {mem_utilization_reward},
            --- scaling_penalty: {scaling_penalty},
            --- scaling_action_reward: {scaling_action_reward},
            --- active_cluster_reward_multiplier: {active_cluster_reward_multiplier}
            
            
            
            """)
            return normalized_reward

        if type == "simple2":
            
            min_replicas_limit = self.limits[agent]['min_replicas']
            max_replicas_limit = self.limits[agent]['max_replicas']
            
            reward = 0
            current_desired_replica_cpu = ceil(num_replicas * (avg_cpu / target_cpu))
            current_desired_replica_mem = ceil(num_replicas * (avg_mem / target_mem))
            
            old_desired_replica_cpu = ceil(last_num_replicas * (last_avg_cpu / target_cpu))
            old_desired_replica_mem = ceil(last_num_replicas * (last_avg_mem / target_mem))
            
            old_best_replica = max(old_desired_replica_cpu, old_desired_replica_mem)
            
            
            
            before_cpu_utilization_reward = self.positive_skew_function(last_avg_cpu, target_cpu)
            before_mem_utilization_reward = self.positive_skew_function(last_avg_mem, target_mem)
            
            after_cpu_utilization_reward = self.positive_skew_function(avg_cpu, target_cpu)
            after_mem_utilization_reward = self.positive_skew_function(avg_mem, target_mem)
            
            total_cpu_utilization_reward = (0.5 * before_cpu_utilization_reward + 0.5 * after_cpu_utilization_reward) * 0.5
            total_mem_utilization_reward = (0.5 * before_mem_utilization_reward + 0.5 * after_mem_utilization_reward) * 0.5
            
            type_of_scale = num_replicas - last_num_replicas
            
            target_cpu = 0.8
            target_mem = 0.8
            cpu_difference_ratio = (avg_cpu - last_avg_cpu) / last_avg_cpu
            mem_difference_ratio = (avg_mem - last_avg_mem) / last_avg_mem
            cpu_similar_threshold = 0.1
            mem_similar_threshold = 0.1
            
            if old_best_replica == old_desired_replica_cpu and old_best_replica == old_desired_replica_mem:
                cpu_weight = 0.5
                mem_weight = 0.5
            elif old_best_replica == old_desired_replica_cpu:
                cpu_weight = 0.8
                mem_weight = 0.2
            elif old_best_replica == old_desired_replica_mem:
                cpu_weight = 0.2
                mem_weight = 0.8
            
            latency50_difference = latency50_ratio - last_latency50_ratio
            latency99_difference = latency99_ratio - last_latency99_ratio
            
            
            if type_of_scale > 0:
                #scale up
                #has CPU and MEM increased or decreased?
                #if cpu_difference > 0 it means either that it was not good.
                #if mem_difference > 0 it means either that it was not good.
                
                if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                    pass
                    #similar ratio as before, cpu_utilization reward doens't need to change
                    
                elif cpu_difference_ratio > cpu_similar_threshold:
                    # With scale up this has increased --> BAD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1-cpu_difference_ratio-cpu_similar_threshold)
                else:
                    #with scale up this has decreased --> GOOD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1+cpu_difference_ratio+cpu_similar_threshold)
                    
                    
                if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                    pass
                    #similar ratio as before, mem_utilization reward doens't need to change
                    
                elif mem_difference_ratio > mem_similar_threshold:
                    # With scale up this has increased --> BAD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1-mem_difference_ratio-mem_similar_threshold)
                    
                else:
                    #with scale up this has decreased --> GOOD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1+mem_difference_ratio+mem_similar_threshold)
                    
            elif type_of_scale < 0:
                #scale down
                #has CPU and MEM increased or decreased?
                #if cpu_difference < 0 it means either that it was not good.
                #if mem_difference < 0 it means either that it was not good.
                
                if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                    pass
                    #similar ratio as before, cpu_utilization reward doens't need to change
                
                elif cpu_difference_ratio > cpu_similar_threshold:
                    # With scale down this has increased --> GOOD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1+cpu_difference_ratio-cpu_similar_threshold)
                else:
                    #with scale down this has decreased --> BAD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1-cpu_difference_ratio+cpu_similar_threshold)
                    
                if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                    pass
                    #similar ratio as before, mem_utilization reward doens't need to change
                    
                elif mem_difference_ratio > mem_similar_threshold:
                    # With scale down this has increased --> GOOD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1+mem_difference_ratio-mem_similar_threshold)
                    
                else:
                    #with scale down this has decreased --> BAD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1-mem_difference_ratio+mem_similar_threshold)
            
            elif type_of_scale == 0:
                #no scale
                #has CPU and MEM increased or decreased?
                #In this case border cases need to be considered.
                #If the difference is 0 it should also represent that we are either at the lower or upper bound of the action possibility and the previous action led us at this point
                
                #cpu and memory rewards are affected by the scale action in general, just not when we are the boundaries and the previous num of replicas is the same as the new one and is the same as one of the two boundaries
                if (num_replicas == min_replicas_limit or num_replicas == max_replicas_limit) and num_replicas == last_num_replicas:
                    #we are at the lower bound and we are not scaling. In this case we check the CPU and Memory utilization and we give a multiplier based on that
                    
                   if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                    total_cpu_utilization_reward = total_cpu_utilization_reward * 2

                    #similar ratio as before, cpu_utilization reward doens't need to change
                    if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                        total_mem_utilization_reward = total_mem_utilization_reward * 2
                     
                else:
                    #we are not at the boundaries and we are not scaling. In this case we check the CPU and Memory utilization and we give a multiplier based on that
                    if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                        pass
                        #similar ratio as before, cpu_utilization reward doens't need to change
                        
                    elif cpu_difference_ratio > cpu_similar_threshold:
                        # With no scale this has increased --> BAD
                        total_cpu_utilization_reward = total_cpu_utilization_reward * (1-cpu_difference_ratio-cpu_similar_threshold)
                    else:
                        #with no scale this has decreased --> GOOD
                        total_cpu_utilization_reward = total_cpu_utilization_reward * (1+cpu_difference_ratio+cpu_similar_threshold)
                        
                    if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                        pass
                        #similar ratio as before, mem_utilization reward doens't need to change
                        
                    elif mem_difference_ratio > mem_similar_threshold:
                        # With no scale this has increased --> BAD
                        total_mem_utilization_reward = total_mem_utilization_reward * (1-mem_difference_ratio-mem_similar_threshold)
                        
                        
                    # Additional reward for maintaining the same state if optimal
            
                    if (abs(avg_cpu - target_cpu) < 0.05 and abs(avg_mem - target_mem) < 0.05):
                        reward += 0.5  # Significant reward for staying optimal
                    elif (abs(avg_cpu - target_cpu) < 0.1 and abs(avg_mem - target_mem) < 0.1):
                        reward += 0.25  # Moderate reward for staying near optimal
                    else:
                        reward -= 0.4  # Penalize if staying but not optimal
                    

            
            resource_usage_weight = 0.7
            latency50_weight = 0.3
            reward = resource_usage_weight * (cpu_weight * total_cpu_utilization_reward + mem_weight * total_mem_utilization_reward) + latency50_weight * (0.5 * latency50_ratio + 0.5 * latency99_ratio)
        
            normalized_reward = np.tanh(reward)
            
            print(f"--- Agent {agent} --- [{self.groups[agent]}] ---")
            print(f"--- Reward: {reward}")
            print(f"--- CPU utilization reward: {total_cpu_utilization_reward}")
            print(f"--- Memory utilization reward: {total_mem_utilization_reward}")
            print(f"--- Latency 50th percentile reward: {latency50_ratio}")
            print(f"--- Latency 99th percentile reward: {latency99_ratio}")
            print(f"--- Normalized reward: {normalized_reward}")
            
            additional_info = dict()
            additional_info['old_best_replica'] = old_best_replica
            additional_info['num_replicas'] = num_replicas
            
            return normalized_reward, additional_info
    
        if type == "simple3":
            #this time with scaling times
            
            cpu_weight = 0.5
            mem_weight = 0.5
            
            if old_best_replica == old_desired_replica_cpu and old_best_replica == old_desired_replica_mem:
                cpu_weight = 0.5
                mem_weight = 0.5
            elif old_best_replica == old_desired_replica_cpu:
                cpu_weight = 0.7
                mem_weight = 0.3
            elif old_best_replica == old_desired_replica_mem:
                cpu_weight = 0.3
                mem_weight = 0.7
            
            
            
            
            if type_of_scale > 0:
                #scale up
                #has CPU and MEM increased or decreased?
                #if cpu_difference > 0 it means either that it was not good.
                #if mem_difference > 0 it means either that it was not good.
                
                if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                    pass
                    #similar ratio as before, cpu_utilization reward doens't need to change
                    
                elif cpu_difference_ratio > cpu_similar_threshold:
                    # With scale up this has increased --> BAD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1-cpu_difference_ratio-cpu_similar_threshold)
                else:
                    #with scale up this has decreased --> GOOD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1+cpu_difference_ratio+cpu_similar_threshold)
                    
                    
                if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                    pass
                    #similar ratio as before, mem_utilization reward doens't need to change
                    
                elif mem_difference_ratio > mem_similar_threshold:
                    # With scale up this has increased --> BAD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1-mem_difference_ratio-mem_similar_threshold)
                    
                else:
                    #with scale up this has decreased --> GOOD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1+mem_difference_ratio+mem_similar_threshold)
                    
                #scaling times
                    
            elif type_of_scale < 0:
                #scale down
                #has CPU and MEM increased or decreased?
                #if cpu_difference < 0 it means either that it was not good.
                #if mem_difference < 0 it means either that it was not good.
                
                if cpu_difference_ratio <= cpu_similar_threshold or cpu_difference_ratio >= -cpu_similar_threshold:
                    pass
                    #similar ratio as before, cpu_utilization reward doens't need to change
                
                elif cpu_difference_ratio > cpu_similar_threshold:
                    # With scale down this has increased --> GOOD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1+cpu_difference_ratio-cpu_similar_threshold)
                else:
                    #with scale down this has decreased --> BAD
                    total_cpu_utilization_reward = total_cpu_utilization_reward * (1-cpu_difference_ratio+cpu_similar_threshold)
                    
                if mem_difference_ratio <= mem_similar_threshold or mem_difference_ratio >= -mem_similar_threshold:
                    pass
                    #similar ratio as before, mem_utilization reward doens't need to change
                    
                elif mem_difference_ratio > mem_similar_threshold:
                    # With scale down this has increased --> GOOD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1+mem_difference_ratio-mem_similar_threshold)
                    
                else:
                    #with scale down this has decreased --> BAD
                    total_mem_utilization_reward = total_mem_utilization_reward * (1-mem_difference_ratio+mem_similar_threshold)
                        
                    

            # scaling_time_penalty = np.tanh(-(1/num_replicas) * (scaling_time)) / 3
            difference_scaling_penalty = 0.15* np.tanh(-difference_action_desired)
            
            
            resource_usage_weight = 0.7
            latency50_weight = 0.3
            #compute reward as a weighted sum of the resource usage and the latency to begin, then we add the penalties and bonuses
            reward = resource_usage_weight * (cpu_weight * total_cpu_utilization_reward + mem_weight * total_mem_utilization_reward) + latency50_weight * (0.5 * latency50_ratio + 0.5 * latency99_ratio)
            reward += difference_scaling_penalty
            # if type_of_scale == 0 :
                # Additional reward for maintaining the same state if optimal
            
            if (abs(avg_cpu - last_avg_cpu) < 0.1 and avg_cpu < target_cpu):
                reward += 0.3  # Significant reward for staying optimal (CPU)
            elif (abs(avg_cpu - last_avg_cpu) < 0.2 and avg_cpu < target_cpu):
                reward += 0.2  # Moderate reward for staying near optimal (CPU)
            elif (abs(avg_cpu - last_avg_cpu) < 0.3 and avg_cpu < target_cpu):
                reward += 0.1  # Small reward for staying close to optimal (CPU)
            else:
                reward -= 0.2  # Penalize if staying but not optimal (CPU)

            if (abs(avg_mem - last_avg_mem) < 0.1 and avg_mem < target_mem):
                reward += 0.3  # Significant reward for staying optimal (Memory)
            elif (abs(avg_mem - last_avg_mem) < 0.2 and avg_mem < target_mem):
                reward += 0.2  # Moderate reward for staying near optimal (Memory)
            elif (abs(avg_mem - last_avg_mem) < 0.3 and avg_mem < target_mem):
                reward += 0.1  # Small reward for staying close to optimal (Memory)
            else:
                reward -= 0.2  # Penalize if staying but not optimal (Memory)
            
            
            normalized_reward = np.tanh(reward)
            
            print(f"--- Agent {agent} --- [{self.groups[agent]}] ---")
            print(f"--- Reward: {reward}")
            print(f"--- CPU utilization reward: {total_cpu_utilization_reward}")
            print(f"--- Memory utilization reward: {total_mem_utilization_reward}")
            print(f"--- Latency 50th percentile reward: {latency50_ratio}")
            print(f"--- Latency 99th percentile reward: {latency99_ratio}")
            print(f"--- Normalized reward: {normalized_reward}")
            
        
        if type == "simple4":
            # Latency reward
            latency_reward = 0.5 * latency50_ratio + 0.5 * latency99_ratio
        

            # Penalize unnecessary scaling actions
            scaling_penalty = -0.5 if type_of_scale != 0 and (abs(avg_cpu - last_avg_cpu) < 0.05 and abs(avg_mem - last_avg_mem) < 0.05) else 0
            
            # Reward for maintaining optimal state
            optimal_state_bonus = 0.5 if (abs(avg_cpu - target_cpu) < 0.1 and abs(avg_mem - target_mem) < 0.1) else 0
            
            # Total reward
            resource_usage_weight = 0.8
            latency_weight = 0.2
            
            reward = (resource_usage_weight * (before_cpu_utilization_reward + before_mem_utilization_reward) + 
                    latency_weight * latency_reward + 
                    scaling_penalty + 
                    optimal_state_bonus)
            
            # print(reward)
            
            normalized_reward = np.tanh(reward)
            
            print(f"--- Agent {agent} ---")
            print(f"--- Reward: {reward}")
            print(f"--- Normalized reward: {normalized_reward}")
            
            
        return normalized_reward, additional_info
    
    def positive_skew_function(self, x, target, alpha=5, beta=2):
        """
        Positive skew function using a beta distribution with a peak at 0.2.
        
        Parameters:
        x (float or np.ndarray): Input value(s) between 0 and 1.
        alpha (float): Shape parameter alpha of the beta distribution.
        beta (float): Shape parameter beta of the beta distribution.
        
        Returns:
        float or np.ndarray: The value of the skew function at x.
        """
        # Ensure x is within the bounds [0, 1]
        if np.any(x < 0):
            raise ValueError("Input value(s) should be over 0")
        
        if np.any(x > 1):
            x = 1
        
        # Compute the beta distribution PDF
        pdf = stats.beta.pdf(x, alpha, beta)
        
        # Normalize the pdf to make sure its maximum value is 1
        max_pdf = stats.beta.pdf((alpha - 1) / (alpha + beta - 2), alpha, beta)
        normalized_pdf = pdf / max_pdf
        
        return normalized_pdf
    
    
    def compute_truncation(self):
        #compute truncation for the environment
        return self.truncated

    def compute_termination(self):
        # compute termination for the environment
        return self.terminated
    
    def set_termination (self):
        #executes a timer that sleeps for a certain amount of time and then sets the termination to True
        self.terminated = True

    def start_termination_timer(self):
        #executes a timer that sleeps for a certain amount of time and then sets the termination to True
        timer = Timer(TIME_TERMINATED, self.set_termination).start()
        


    def compute_infos(self, observations, reward, terminated, agent):
        pass


    def check_app_ready(self):
        # check if the application is ready
        #is the app deployed ? Check with K8s
        # try:
        #     resp = self.appsV1.read_namespaced_deployment(name=self.app_name, namespace=self.app_namespace)
        #     if resp.status.ready_replicas == 0 or resp.status.available_replicas == 0:
        #         return False
        # except Exception as e:
            
        #     return False
        
        #is the app ready? Check with K8s that every pod is actually ready and don't go out until they are
        try:
            while True:
                resp = self.coreV1.list_namespaced_pod(namespace=self.app_namespace, label_selector=f"app={self.app_name}")
                ready_pods = [pod for pod in resp.items if pod.status.phase == 'Running' and
                              any(cond.type == 'Ready' and cond.status == 'True' for cond in pod.status.conditions)]
                terminating_pods = [pod for pod in resp.items if pod.metadata.deletion_timestamp is not None]

                if len(ready_pods) == len(resp.items) and len(terminating_pods) == 0:
                    print("App ready")
                    break
                time.sleep(TIME_API_CHECK_REPLICAS)
        except Exception as e:
            print("App is not ready or there is some error in the pods creation")
            return False
    
        
        return True