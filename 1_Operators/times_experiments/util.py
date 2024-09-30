import ast

from kubernetes import client, watch
import threading
import os

##########


# constants

HPA_DOMAIN = 'autoscaling.k8s.io'
HPA_VERSION = 'v1'
HPA_PLURAL = 'horizontalpodautoscalers'

INITIAL_CPU_LIMIT = 1024  # millicore
INITIAL_MEMORY_LIMIT = 2048  # MiB
INITIAL_NUM_REPLICAS = 1

MIN_REPLICAS = 1
MAX_REPLICAS = 10
MIN_CPU_LIMIT = 100  # millicore
MAX_CPU_LIMIT = 2048  # millicore
MIN_MEMORY_LIMIT = 256  # MiB
MAX_MEMORY_LIMIT = 3078  # MiB

ACTION_INTERVAL = 10  # OPERATOR REQUEST INTERVAL FOR PAs
ACTION_STOP_INTERVAL = 6

TIME_PRECISION = 0.01  # second


##########  OPERATOR CLASSES UTILS ##########


# class OperatorConfigs:
class OperatorConfigs:
    def __init__(self, app_name="teastore", namespace="default", group_file='experiments_group/exn.txt'):
        self.app_name = app_name
        self.namespace = namespace
        self.run_states = {}
        self.group_states = []
        self.group_file = group_file
        self.group_desired_replicas = {}  # map for desired replicas for each group
        self.groups = []
        self.number_of_tries = 0
        self.count_of_tries = 0
        # APIs
        self.coreV1 = client.CoreV1Api()
        self.appsV1 = client.AppsV1Api()
        self.customObjects = client.CustomObjectsApi()
        self.number_of_nodes = 0

    def add_run(self, run):
        self.run_states[run] = ScalingStates()

    def change_run_replicas(self, run, num_replicas):
        if run in self.run_states:
            self.run_states[run].num_replicas = num_replicas

    def add_run_replicas(self, run, num_replicas):
        if run in self.run_states:
            self.run_states[run].num_replicas += num_replicas

    def read_groups(self, group_file='experiments_group/exn.txt'):
        print("Trying to read group")
        # Check if file exists
        if os.path.exists(group_file):
            with open(group_file, 'r') as file:
                print("Reading groups")
                first_line = file.readline().strip()

                # Parse the first line
                parts = first_line.split(';')
                print(parts)
                self.number_of_tries = int(parts[0])
                replicas_list = ast.literal_eval(parts[1])  # Assuming this is safe to use in your controlled environment

                # Read the replicas for each microservice
                group_index = 0
                for line in file:
                    line = line.strip()
                    if line:  # Ensure the line is not empty
                        self.groups.append(line.split(','))
                        self.group_desired_replicas[group_index] = replicas_list[group_index]
                        group_index += 1

            print("Groups read")
            print(f"Number of tries: {self.number_of_tries}")
            print(f"Replicas: {self.group_desired_replicas}")
            print(f"Groups: {self.groups}")
        else:
            print("Group file does not exist")

    def create_groups_numeric_schema(self, groups: list):
        # could put more sofisticated logic here
        # for now we base it off a static schema given from an environment variable,
        # supposing the user knows how to the application is deployed and how he wants the groups to be deployed
        # like 7 deployments in a 3, 3, 2 schema
        # schema should be an array of ints
        start_index = 0
        for group_size in groups:
            self.group_states.append(
                [self.run_states[key] for key in list(self.run_states.keys())[start_index: start_index + group_size]])
            start_index += group_size


class ScalingStates:
    def __init__(self):
        self.num_replicas = INITIAL_NUM_REPLICAS
        # self.cpu_limit = INITIAL_CPU_LIMIT
        # self.memory_limit = INITIAL_MEMORY_LIMIT

        # modify these when scaling


thread_daemon_dict = {}
immutable_keys = set()

lock = threading.Lock()


def modify_thread_dict(group, thread_id):
    with lock:
        if group not in immutable_keys:
            thread_daemon_dict[group] = thread_id
            immutable_keys.add(group)


def remove_thread_dict(group):
    with lock:
        if group in thread_daemon_dict:
            del thread_daemon_dict[group]
            immutable_keys.remove(group)


def get_thread_dict(group):
    with lock:
        return thread_daemon_dict.get(group)


class scaling_times:
    def __init__(self):
        self.time_before_api = 0
        self.time_after_api = 0
        self.time_last_scaled_ready = 0
        self.time_total = 0
        self.replicas = 0

##########  RLENV CLASSES UTILS ##########
