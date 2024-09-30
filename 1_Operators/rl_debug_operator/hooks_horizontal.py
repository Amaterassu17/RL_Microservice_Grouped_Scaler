import random
import kopf
from kubernetes import client, config, watch

import sys
import yaml
import argparse
import asyncio
import time
import socket
import os
import json
import threading
from util import *
import logging

# rest of the code...





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


@kopf.daemon('apps', 'v1', 'deployments')
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

        if selected_group is not None:
            if get_thread_dict(selected_group) is None:
                modify_thread_dict(selected_group, run)
            
        if selected_group is not None and get_thread_dict(selected_group) == run:  
            print(f"Scaling every {ACTION_INTERVAL} seconds for group {selected_group}")
            while not stopped:
                stopped.wait(ACTION_INTERVAL)
                memo["timestamps"] = {}
                for microservice in config_operator.groups[selected_group]:
                    if microservice in config_operator.run_states.keys():
                        memo["timestamps"][microservice] = scaling_times()
                scaling_random_group(selected_group, config_operator, memo)
                export_scaling_times(memo)

        else:
            print("Killing daemon")
    else:
        print("Not the desired app")



def scaling_random_group(group, config_operator, memo):
    try:
            
        running_for_group = 0
        num_replicas = random.randint(MIN_REPLICAS, MAX_REPLICAS)
        print(f"Scaling {config_operator.groups[group]} to {num_replicas} replicas")
        for microservice in config_operator.groups[group]:
            if microservice in config_operator.run_states.keys():
                try:
                    running_for_group += 1
                    memo["timestamps"][microservice].time_before_api = time.time()
                    print(memo["timestamps"][microservice].time_before_api)
                    resp = config_operator.appsV1.patch_namespaced_deployment_scale(
                        name=microservice, namespace=config_operator.namespace, body={"spec": {"replicas": num_replicas}}
                        )
                    memo["timestamps"][microservice].time_after_api = time.time()
                    memo["timestamps"][microservice].replicas = num_replicas
                    print(memo["timestamps"][microservice].time_after_api)

 
                except Exception as e:
                    print(f"Problem in scaling: {e}")
            

        groups_copy = config_operator.groups[group][:]
        while len(groups_copy) > 0 and running_for_group > 0:
                time.sleep(TIME_PRECISION)
                for microservice in groups_copy:
                    if microservice in config_operator.run_states.keys():
                        try:
                            resp = config_operator.appsV1.read_namespaced_deployment_status(name=microservice, namespace=config_operator.namespace)
                            ready_replicas = resp.status.ready_replicas
                            if ready_replicas == num_replicas:
                                print(f"Deployment {microservice} is ready")
                                memo["timestamps"][microservice].time_last_scaled_ready = time.time()
                                memo["timestamps"][microservice].time_total = memo["timestamps"][microservice].time_last_scaled_ready - memo["timestamps"][microservice].time_before_api
                                groups_copy.remove(microservice)
                                print(f"Time taken for {microservice}: {memo['timestamps'][microservice].time_total}")
                                break
                        except Exception as e:
                            print(f"Problem in reading replicas: {e}")
    except Exception as e:
        print(f"Problem in scaling: {e}")



def export_scaling_times(memo):
    #if it does exist append after a \n, otherwise create the file
    #format is microservice, time_before_api, time_after_api, time_last_scaled_ready, time_total
    #if created for the first time write the header
    try:
        with open("scaling_times.csv", "a") as f:
            if os.stat("scaling_times.csv").st_size == 0:
                f.write("Microservice, Time before API, Time after API, Time last scaled ready, Time total, Replicas\n")
            for microservice in memo["timestamps"]:
                f.write(f"{microservice}, {memo['timestamps'][microservice].time_before_api}, {memo['timestamps'][microservice].time_after_api}, {memo['timestamps'][microservice].time_last_scaled_ready}, {memo['timestamps'][microservice].time_total}, {memo['timestamps'][microservice].replicas}\n")
            
            f.write("\n")
    except Exception as e:
        print(f"Problem in exporting scaling times: {e}")

    memo["timestamps"] = {}


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
