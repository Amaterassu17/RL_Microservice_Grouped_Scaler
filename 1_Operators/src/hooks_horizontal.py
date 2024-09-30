import random
import kopf
from kubernetes import client, config
import sys
import yaml
import argparse
import asyncio
import time
import socket
import os
import json
from flask import Flask, request, jsonify
from collections import defaultdict
import threading
import logging



##########


# constant

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

ACTION_INTERVAL = 60  # OPERATOR REQUEST INTERVAL FOR PAs


# constant FLASK
# app = Flask(__name__)
# HOST="localhost"
# PORT=5000

# constant K8S

   

# Prometheus vars if needed
# system metrics are exported to system-space Prometheus instance
# PROM_URL = 'http://localhost:9090'
# PROM_TOKEN = None  # if required, to be overwritten by the environment variable


# # custom metrics are exported to user-space Prometheus instance
# PROM_URL_USERSPACE = 'http://localhost:9090'
# # PROM_URL_USERSPACE = 'https://prometheus-k8s.openshift-monitoring.svc.cluster.local:9091'  # used when running the RL controller as a pod in the cluster
# PROM_TOKEN_USERSPACE = None  # user-space Prometheus does not require token access

# FORECASTING_SIGHT_SEC = 30        # look back for 30s for Prometheus data
# HORIZONTAL_SCALING_INTERVAL = 10  # wait for 10s for horizontal scaling to update
# VERTICAL_SCALING_INTERVAL = 10    # wait for 10s for vertical scaling to update


##########


# class OperatorConfigs:
class OperatorConfigs:
    def __init__(self, app_name="teastore", namespace="default", group_file="./groups.txt"):
        self.app_name = app_name
        self.namespace = namespace
        self.run_states = {}
        self.group_states = []
        self.group_desired_replicas = []  # map for desired replicas for each group
        self.group_file= group_file
        self.groups = []
        #APIs
        self.coreV1 = client.CoreV1Api()
        self.appsV1 = client.AppsV1Api()
        self.customObjects = client.CustomObjectsApi()
        
    
        self.read_groups()

    def add_run(self, run):
        self.run_states[run] = ScalingStates()

    def change_run_replicas(self, run, num_replicas):
        if run in self.run_states:
            self.run_states[run].num_replicas = num_replicas

    def add_run_replicas(self, run, num_replicas):
        if run in self.run_states:
            self.run_states[run].num_replicas += num_replicas

    def read_groups(self):
        print("Trying to read group")
        #check if file exists
        if os.path.exists(self.group_file):
            with open(self.group_file, 'r') as file:
                print("Reading groups")
                for line in file:
                    elements= line.strip().split(',')
                    self.groups.append(elements)
            
            print("Groups read")
            print(self.groups)
        else:
            print("Group file does not exist")
        
    


    def create_groups_numeric_schema(self, groups: list):
        #could put more sofisticated logic here
        #for now we base it off a static schema given from an environment variable, 
        #supposing the user knows how to the application is deployed and how he wants the groups to be deployed
        #like 7 deployments in a 3, 3, 2 schema
        #schema should be an array of ints
        start_index=0
        for group_size in groups:
            self.group_states.append([self.run_states[key] for key in list(self.run_states.keys())[start_index: start_index+group_size]])
            start_index+=group_size
    

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

    




#########






@kopf.on.login()
def login_fn(**kwargs):
    print("Logging in...")
    
    return kopf.login_via_client(**kwargs)

@kopf.on.startup()
def startup_fn(memo: kopf.Memo,**kwargs):
    print("Starting up...")

    
    if str(os.getenv('KUBERNETES_POD')).lower().strip() == "true":
            print("using kube pod config")
            config.load_incluster_config() #works well in a pod, No bearer token please}")
    else:
            print("Using kubeconfig")
            config.load_kube_config() 


    config_operator = OperatorConfigs()
    
    if os.getenv('APP_NAME') is not None:
        config_operator.app_name = os.getenv('APP_NAME')

    if os.getenv('NAMESPACE') is not None:
        config_operator.namespace = os.getenv('NAMESPACE')

    if os.getenv('GROUP_FILE') is not None:
        print("Group file is set")
        config_operator.group_file = os.getenv('GROUP_FILE')

    memo["config_operator"] = config_operator
    print("Config Initialized")



# @kopf.on.create('apps', 'v1', 'deployments')
# def app_create_handler(spec, **kwargs):

#     app = kwargs['body']
#     metadata = app['metadata']
#     labels = metadata.get('labels', {})
#     app_name = labels.get('app')
    
#     if str(app_name).strip() == str(config_operator.app_name).strip():
#         config_operator.add_run(app_name)

        # retrieve deployment's metadata.name

        
    


@kopf.on.delete('apps', 'v1', 'deployments')
def app_delete_handler(spec, **kwargs):

    app = kwargs['body']
    metadata = app['metadata']
    labels = metadata.get('labels', {})
    app_name = labels.get('app')
    print("App name:", app_name)
    print("config_operator.app_name:", config_operator.app_name)

    if str(app_name).strip() == str(config_operator.app_name).strip():
        print("Deleting app")
        config_operator.run_states.pop(app_name)
        print("Deleted app")
        

@kopf.on.cleanup()
def cleanup_fn(memo: kopf.Memo, **kwargs):
    config_operator = memo["config_operator"]
    print("Cleaning up...")
    # delete all the PAs
    # patch removing finalizers in all the microservices contained in the keys of config_operator.run_states
    for run in config_operator.run_states.keys():
        try:
            resp=config_operator.appsV1.patch_namespaced_deployment(name=run, namespace=config_operator.namespace, body={"metadata": {"finalizers": []}})
            print(f"Deleted finalizers for {run}")
    
        except Exception as e:
            print(f"Problem in deleting finalizers: {e}")
    print("Cleaned up")
    return {'message': 'cleaned up'}


@kopf.daemon('apps', 'v1', 'deployments') #when creating a deployment in apps/v1 execute function
def daemon_sync_scaling(memo: kopf.Memo ,stopped, **kwargs):
    config_operator = memo["config_operator"]
    app = kwargs['body']
    metadata = app['metadata']
    run = metadata.get('name')
    labels = metadata.get('labels', {})
    app_name = labels.get('app')
    label_selector = f"app={config_operator.app_name}"

    if app_name == config_operator.app_name:

        config_operator.add_run(run)
        
        selected_group = None
        for index, inner_array in enumerate(config_operator.groups):
            if run in inner_array:
                selected_group = index
                break

        #trying using the shared dict
        if selected_group is not None:
            if get_thread_dict(selected_group) is None:
                modify_thread_dict(selected_group, run)
            

        if selected_group is not None and get_thread_dict(selected_group) == run:  
            print(f"Scaling every {ACTION_INTERVAL} seconds for group {selected_group}")
            while not stopped:
                time.sleep(ACTION_INTERVAL)
                try:
                    num_replicas = random.randint(MIN_REPLICAS, MAX_REPLICAS)
                    print(f"Scaling {config_operator.groups[selected_group]} to {num_replicas} replicas")
                    for microservice in config_operator.groups[selected_group]:
                        try:
                            resp = config_operator.appsV1.patch_namespaced_deployment_scale(name=microservice, namespace=config_operator.namespace, body={"spec": {"replicas": num_replicas}})
                        except Exception as e:
                            print(f"Problem in scaling: {e}")
                except:
                    # In case crashes command
                    # kubectl patch deployments teastore-auth teastore-db teastore-image teastore-persistence teastore-image teastore-webui teastore-recommender teastore-registry -p '{"metadata": {"finalizers": []}}' --type merge
                    print("Problem in scaling")
        # else:
        #     print("Daemon already started for this group")
    else:
        print("Not the desired app")





# Functions for scaling


def hpa_add_num_replicas(hpa_name, namespace, delta_num_replicas):
    # TODO patch HPA deployment
    print("Adding limits to HPA")


def hpa_set_num_replicas(hpa_name, namespace, num_replicas):
    # TODO patch HPA deployment
    print("Setting limits to HPA")





# def mpa_sanity_check(action, run, namespace):
#     mpa_state = config_operator.mpa_states[run, namespace]
#     print(f"Sanity check for {run}")
#     print(f"Initial MPA state: {mpa_state}")
#     if mpa_state is None:
#         print(f"MPA state for {run} does not exist")
#         return False

#     if action['horizontal'] != 0:
#         if mpa_state.num_replicas + action['horizontal'] < MIN_REPLICAS or mpa_state.num_replicas + action['horizontal'] > MAX_REPLICAS:
#             print(f"Horizontal scaling action for {run} out of bounds")
#             return False
#     elif action['vertical_cpu'] != 0:
#         if mpa_state.cpu_limit + action['vertical_cpu'] < MIN_CPU_LIMIT or mpa_state.cpu_limit + action['vertical_cpu'] > MAX_CPU_LIMIT:
#             print(f"Vertical scaling action for {run} out of bounds")
#             return False
#     elif action['vertical_memory'] != 0:
#         if mpa_state.memory_limit + action['vertical_memory'] < MIN_MEMORY_LIMIT or mpa_state.memory_limit + action['vertical_memory'] > MAX_MEMORY_LIMIT:
#             print(f"Vertical scaling action for {run} out of bounds")
#             return False
#     return True


# def test_mpa_num_replicas_random(mpa_name):
#     if (config_operator.pa_created is True):
#         random_num_replicas = random.randint(MIN_REPLICAS, MAX_REPLICAS)
#         print(
#             f"Random number of replicas for {mpa_name}: {random_num_replicas}")
#         mpa_set_num_replicas(
#             mpa_name, config_operator.namespace, random_num_replicas)


# @kopf.daemon('v1', 'pods')
# def pod_daemon_handler(stopped ,**kwargs):
#     print("Starting asynch daemon")
#     while not stopped:

#         if(received_json is not None):
#             print("Received json")
#             print(received_json)
#             received_json = None


#         time.sleep(ACTION_INTERVAL)
#         # print("Checking for pods")
#         # pods = v1.list_pod_for_all_namespaces(watch=False)
#         # for pod in pods.items:
#         #     labels = pod.metadata.labels
#         #     if labels.get('app') == app_name:
#         #         print(f"Pod for app: {app_name} exists")
#         #         break
#         # else:
#         #     print(f"Pod for app: {app_name} does not exist")
#         #     break


# def patch_mpa_deployment(run, namespace, patch):
#     # patch MPA deployment
#     mpa_name = f"{run}-mpa"


#     # patch = {
#     #     "spec": {
#     #         "constraints": {
#     #             "minReplicas": 1,
#     #             "maxReplicas": 10
#     #         },
#     #         "metrics": [
#     #             {
#     #                 "type": "Resource",
#     #                 "resource": {
#     #                     "name": "cpu",
#     #                     "target": {
#     #                         "type": "Utilization",
#     #                         "averageUtilization": 50
#     #                     }
#     #                 }
#     #             },
#     #             {
#     #                 "type": "Resource",
#     #                 "resource": {
#     #                     "name": "memory",
#     #                     "target": {
#     #                         "type": "Utilization",
#     #                         "averageUtilization": 50
#     #                     }
#     #                 }
#     #             }
#     #         ]
#     #     }
#     # }
#     customObjectApi.patch_namespaced_custom_object(group="autoscaling.k8s.io", version="v1alpha1", namespace=namespace, plural="multidimpodautoscalers", name=mpa_name, body=patch)


# # set the vertical scaling recommendation to MPA
#     def set_vertical_scaling_recommendation(self, cpu_limit, memory_limit):
#         # update the recommendations
#         container_recommendation = {"containerName": "", "lowerBound": {}, "target": {}, "uncappedTarget": {}, "upperBound": {}}
#         container_recommendation["lowerBound"]['cpu'] = str(cpu_limit) + 'm'
#         container_recommendation["target"]['cpu'] = str(cpu_limit) + 'm'
#         container_recommendation["uncappedTarget"]['cpu'] = str(cpu_limit) + 'm'
#         container_recommendation["upperBound"]['cpu'] = str(cpu_limit) + 'm'
#         container_recommendation["lowerBound"]['memory'] = str(memory_limit) + 'Mi'
#         container_recommendation["target"]['memory'] = str(memory_limit) + 'Mi'
#         container_recommendation["uncappedTarget"]['memory'] = str(memory_limit) + 'Mi'
#         container_recommendation["upperBound"]['memory'] = str(memory_limit) + 'Mi'

#         recommendations = []
#         containers = self.get_target_containers()
#         for container in containers:
#             vertical_scaling_recommendation = container_recommendation.copy()
#             vertical_scaling_recommendation['containerName'] = container
#             recommendations.append(vertical_scaling_recommendation)

#         patched_mpa = {"recommendation": {"containerRecommendations": recommendations}, "currentReplicas": self.states['num_replicas'], "desiredReplicas": self.states['num_replicas']}
#         body = {"status": patched_mpa}
#         mpa_api = client.CustomObjectsApi()

#         # Update the MPA object
#         # API call doc: https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CustomObjectsApi.md#patch_namespaced_custom_object
#         try:
#             mpa_updated = mpa_api.patch_namespaced_custom_object(group=MPA_DOMAIN, version=MPA_VERSION, plural=MPA_PLURAL, namespace=self.mpa_namespace, name=self.mpa_name, body=body)
#             print("Successfully patched MPA object with the recommendation: %s" % mpa_updated['status']['recommendation']['containerRecommendations'])
#         except ApiException as e:
#             print("Exception when calling CustomObjectsApi->patch_namespaced_custom_object: %s\n" % e)

#     # execute the action after sanity check
#     def execute_action(self, action):
#         if action['vertical_cpu'] != 0:
#             # vertical scaling of cpu limit
#             self.states['cpu_limit'] += action['vertical_cpu']
#             self.set_vertical_scaling_recommendation(self.states['cpu_limit'], self.states['memory_limit'])
#             # sleep for a period of time to wait for update
#             time.sleep(VERTICAL_SCALING_INTERVAL)
#         elif action['vertical_memory'] != 0:
#             # vertical scaling of memory limit
#             self.states['memory_limit'] += action['vertical_memory']
#             self.set_vertical_scaling_recommendation(self.states['cpu_limit'], self.states['memory_limit'])
#             # sleep for a period of time to wait for update
#             time.sleep(VERTICAL_SCALING_INTERVAL)
#         elif action['horizontal'] != 0:
#             # scaling in/out
#             num_replicas = self.states['num_replicas'] + action['horizontal']
#             self.api_instance.patch_namespaced_deployment_scale(
#                 self.app_name,
#                 self.app_namespace,
#                 {'spec': {'replicas': num_replicas}}
#             )
#             print('Scaled to', num_replicas, 'replicas')
#             self.states['num_replicas'] = num_replicas
#             # sleep for a period of time to wait for update
#             time.sleep(HORIZONTAL_SCALING_INTERVAL)
#         else:
#             # no action to perform
#             print('No action')
#             pass


# @app.route('/api/rl-operator', methods=['POST'])
# def receive_json():
#     data = request.get_json()
#     print(data)
#     received_json = jsonify(data)
#     return received_json
