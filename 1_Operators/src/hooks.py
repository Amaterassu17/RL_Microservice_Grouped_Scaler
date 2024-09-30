import kopf
from kubernetes import client, config
import sys
import yaml
import argparse
import asyncio
import time
import socket
import json
from flask import Flask, request, jsonify

#constant
MPA_DOMAIN = 'autoscaling.k8s.io'
MPA_VERSION = 'v1alpha1'
MPA_PLURAL = 'multidimpodautoscalers'
INITIAL_CPU_LIMIT = 1024 # millicore
INITIAL_MEMORY_LIMIT = 2048 # MiB
INITIAL_NUM_REPLICAS = 1

ACTION_INTERVAL = 5 # OPERATOR REQUEST INTERVAL FOR PAs


#constant FLASK
app = Flask(__name__)
HOST="localhost"
PORT=5000

#constant K8S

# config.load_kube_config() works well not in a pod but run as a python script from the node

config.load_incluster_config() #works well in a pod

v1 = client.CoreV1Api()
customObjectApi=client.CustomObjectsApi()
podAutoscalerApi=client.AutoscalingV2Api()

#Prometheus vars if needed
# system metrics are exported to system-space Prometheus instance
PROM_URL = 'http://localhost:9090' 
PROM_TOKEN = None  # if required, to be overwritten by the environment variable


#class OperatorConfigs:
class OperatorConfigs:
    def __init__(self, app_name="teastore", autoscale_strategy="mpa", namespace="default"):
        self.app_name = app_name
        self.namespace = namespace
        self.autoscale_strategy = autoscale_strategy
        self.run_list = []
        self.pa_list = []
        self.pod_list = []
        self.pa_created = False
        #map for current mpa cpu, memory and number of replicas
        self.mpa_states = {}


    def add_mpa_state(self, run, namespace, num_replicas, cpu_limit, memory_limit):
        self.mpa_states[run,namespace] = MPAStates()
        self.mpa_states[run,namespace].num_replicas = num_replicas
        self.mpa_states[run,namespace].cpu_limit = cpu_limit
        self.mpa_states[run,namespace].memory_limit = memory_limit

    def get_mpa_state(self, run, namespace):
        return self.mpa_states[run,namespace]
    
    def update_mpa_state(self, run, namespace, num_replicas, cpu_limit, memory_limit):
        self.mpa_states[run,namespace].num_replicas = num_replicas
        self.mpa_states[run,namespace].cpu_limit = cpu_limit
        self.mpa_states[run,namespace].memory_limit = memory_limit

    def update_num_replicas(self, run, namespace, num_replicas):
        self.mpa_states[run,namespace].num_replicas = num_replicas

    def update_cpu_limit(self, run, namespace, cpu_limit):
        self.mpa_states[run,namespace].cpu_limit = cpu_limit

    def update_memory_limit(self, run, namespace, memory_limit):
        self.mpa_states[run,namespace].memory_limit = memory_limit
        

#class MPA States
class MPAStates:
    def __init__(self):
        self.num_replicas= INITIAL_NUM_REPLICAS
        self.cpu_limit= INITIAL_CPU_LIMIT
        self.memory_limit= INITIAL_MEMORY_LIMIT

        #modify these when scaling





# custom metrics are exported to user-space Prometheus instance
PROM_URL_USERSPACE = 'http://localhost:9090'
# PROM_URL_USERSPACE = 'https://prometheus-k8s.openshift-monitoring.svc.cluster.local:9091'  # used when running the RL controller as a pod in the cluster
PROM_TOKEN_USERSPACE = None  # user-space Prometheus does not require token access

# FORECASTING_SIGHT_SEC = 30        # look back for 30s for Prometheus data
# HORIZONTAL_SCALING_INTERVAL = 10  # wait for 10s for horizontal scaling to update
# VERTICAL_SCALING_INTERVAL = 10    # wait for 10s for vertical scaling to update




config_operator = OperatorConfigs()
@kopf.on.login()
def login_fn(**kwargs):
    print("Logging in...")
    print("Initializing config")
    #config # CHANGE STUFF

    # config_operator.app_name = "teastore"
    # config_operator.autoscale_strategy = "mpa"
    # config_operator.namespace = "default"

    print("Initialized config")
    app.run(host=HOST, port=PORT)
    print("App running")
    #print config
    # print(config.list_kube_config_contexts())
    return kopf.login_via_client(**kwargs)










# @kopf.on.create('kopfexamples')
# def create_fn(spec, **kwargs):
#     print(f"And here we are! Creating: {spec}")
#     return {'message': 'hello world'}  # will be the new status

@kopf.on.create('v1', 'pods')
def pod_create_handler(spec, **kwargs):

    pod=kwargs['body']
    labels=pod.metadata.labels

    if labels.get('app') == config_operator.app_name:
        #retrieve deployment's metadata.name
        run=labels.get('run')
        if(run not in run_list):
            config_operator.run_list.append(run)
            config_operator.pod_list.append(pod.metadata.name)
            config_operator.pa_list.append(f"{run}-{autoscale_strategy}")
            #HORIZONTAL VERTICAL NOT WORKING, MULTI WORKING WITH CRDs
            if(config_operator.autoscale_strategy == 'hpa'):
                #create HPA deployment
                deployment = {
                    "apiVersion": "autoscaling/v2",
                    "kind": "HorizontalPodAutoscaler",
                    "metadata": {
                        "name": f"{run}-hpa",
                        "namespace": pod.metadata.namespace
                    },
                    "spec": {
                        "scaleTargetRef": {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": f"{run}"
                        },
                        "minReplicas": 1,
                        "maxReplicas": 10,
                        "metrics": [
                            {
                                "type": "Resource",
                                    "resource": {
                                        "name": "cpu",
                                        "target": {
                                            "type": "Utilization",
                                            "averageUtilization": 50
                                        }
                                    }

                            }
                        ]
                    }
                }
                podAutoscalerApi.create_namespaced_horizontal_pod_autoscaler(body=deployment, namespace=pod.metadata.namespace)
            elif(config_operator.autoscale_strategy == 'vpa'):
                #create VPA deployment
                deployment = {
                    "apiVersion": "autoscaling.k8s.io/v1",
                    "kind": "VerticalPodAutoscaler",
                    "metadata": {
                        "name": f"{run}-vpa",
                        "namespace": pod.metadata.namespace
                    },
                    "spec": {
                        "targetRef": {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": f"{run}-mpa"
                        },
                        "updatePolicy": {
                            "updateMode": "Auto"
                        }
                    }
                }
                customObjectApi.create_namespaced_custom_object(group="autoscaling.k8s.io", version="v1", namespace=pod.metadata.namespace, plural="verticalpodautoscalers", body=deployment)
            elif(config_operator.autoscale_strategy == 'mpa'):
            #create MPA deployment
                deployment = {
                    "apiVersion": "autoscaling.k8s.io/v1alpha1",
                    "kind": "MultidimPodAutoscaler",
                    "metadata": {
                        "name": f"{run}-mpa",
                        "namespace": pod.metadata.namespace
                    },
                    "spec": {
                        "scaleTargetRef": {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": f"{run}"
                        },
                        "resourcePolicy": {
                            "containerPolicies": [
                                {
                                    "containerName": '*',
                                    "minAllowed": {
                                        "cpu": "100m",
                                        "memory": "50Mi"
                                    },
                                    "maxAllowed": {
                                        "memory": "500Mi"
                                    },
                                    "controlledResources": ["cpu", "memory"]
                                }
                            ]
                        },
                        "constraints": {
                            "minReplicas": 1,
                            "maxReplicas": 10
                        },
                        "metrics": [
                            {
                                "type": "Resource",
                                "resource": {
                                    "name": "cpu",
                                    "target": {
                                        "type": "Utilization",
                                        "averageUtilization": 50
                                    }
                                }
                            },
                            {
                                "type": "Resource",
                                "resource": {
                                    "name": "memory",
                                    "target": {
                                        "type": "Utilization",
                                        "averageUtilization": 50
                                    }
                                }
                            }
                        ]
                    }
            }
                customObjectApi.create_namespaced_custom_object(group="autoscaling.k8s.io", version="v1alpha1", namespace=pod.metadata.namespace, plural="multidimpodautoscalers", body=deployment)
        
            config_operator.pa_created = True
        else:
            print(f"Pod for app: {config_operator.app_name} already exists")

    
    if labels.get('app') == "mpa-operator":
        print("MPA Operator pod created")


@kopf.on.delete('v1', 'pods')
def pod_delete_handler(spec, **kwargs):
    # print(f'Deleting: {spec}')
    pod = kwargs['body']
    labels = pod.metadata.labels

    if labels.get('app') == config_operator.app_name:
        # retrieve deployment's metadata.name
        run = labels.get('run')

        if run in config_operator.run_list:
            config_operator.run_list.remove(run)
            config_operator.pod_list.remove(pod.metadata.name)
            config_operator.pa_list.remove(f"{run}-{config_operator.autoscale_strategy}")
            if config_operator.autoscale_strategy == 'hpa':
                # delete HPA deployment
                hpa_name = f"{run}-hpa"
                podAutoscalerApi.delete_namespaced_horizontal_pod_autoscaler(name=hpa_name, namespace=pod.metadata.namespace)
            elif config_operator.autoscale_strategy == 'vpa':
                # delete VPA deployment
                vpa_name = f"{run}-vpa"
                customObjectApi.delete_namespaced_custom_object(group="autoscaling.k8s.io", version="v1", namespace=pod.metadata.namespace, plural="verticalpodautoscalers", name=vpa_name)
            elif config_operator.autoscale_strategy == 'mpa':
            # delete MPA deployment
                mpa_name = f"{run}-mpa"
                customObjectApi.delete_namespaced_custom_object(group="autoscaling.k8s.io", version="v1alpha1", namespace=pod.metadata.namespace, plural="multidimpodautoscalers", name=mpa_name)
            pa_created = False
        else:
            print(f"Pod for app: {config_operator.app_name} does not exist")
        # print(f"Deleting pod for app: {app_name}")


def mpa_change_num_replicas(run,namespace,delta_num_replicas):
    
    #change state
    config_operator.update_num_replicas(run,namespace,config_operator.get_mpa_state(run,namespace).num_replicas+delta_num_replicas)
    #patch MPA deployment



# def mpa_set_num_replicas(run,namespace,num_replicas):
    





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
