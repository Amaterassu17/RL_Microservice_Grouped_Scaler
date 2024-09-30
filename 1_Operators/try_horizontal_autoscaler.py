import random
from kubernetes import client, config
import sys
from kubernetes.client import configuration
from kubernetes.client.rest import ApiException
import yaml
import argparse
import asyncio
import time
import os
from pick import pick

MIN_REPLICAS=1
MAX_REPLICAS=10
ACTION_TIME=120

CPU_UTILIZATION_THRESHOLD = 80
MEMORY_UTILIZATION_THRESHOLD = 80



def main():
    print("Start")

    parser = argparse.ArgumentParser()
    parser.add_argument('--app_name', type=str, default='teastore')
    parser.add_argument('--app_namespace', type=str, default='default')
    #add other things

    options = parser.parse_args()
    app_name = options.app_name
    app_namespace = options.app_namespace

    print(f"app_name: {app_name}")
    print(f"app_namespace: {app_namespace}")

    run_list=[]

    try:
        contexts, active_context = config.list_kube_config_contexts()
        if not contexts:
            print("Cannot find any context in kube-config file")
            return
        contexts = [context['name'] for context in contexts]
        active_index = contexts.index(active_context['name'])
        options, _ = pick(contexts, title="Pick the context to load", default_index= active_index)
        
        
        # config.load_kube_config(context=options)
        # config.load_incluster_config()

        config.load_kube_config()
        print(f"Active host is {configuration.Configuration().host}")


        coreV1Api = client.CoreV1Api()
        customObjectsApi = client.CustomObjectsApi()
        autoscalingV1Api = client.AutoscalingV1Api()
    
        appsV1Api = client.AppsV1Api()
    
        #try node printing
        # resp = coreV1Api.list_node()
        # print(resp)

        # get apps to scale
        label_selector = f"app={app_name}"
        try:
            resp=appsV1Api.list_namespaced_deployment(namespace=app_namespace, label_selector=label_selector)
            for i in resp.items:
                print(i.metadata.name)
                run_list.append(i.metadata.name)
        except:
            print("Problem in retrieving/printing apps")


        #put limits in the deployments if not already present.
        try:
            if len(run_list) > 0:
                for i in run_list:
                    resp = appsV1Api.read_namespaced_deployment(name=i, namespace=app_namespace)
                    if not resp.spec.template.spec.containers[0].resources or resp.spec.template.spec.containers[0].resources.limits is None:
                        print(f"Adding limits to {i}")
                        resp.spec.template.spec.containers[0].resources = client.V1ResourceRequirements(
                            limits={"cpu": "500m", "memory": "512Mi"},
                            requests={"cpu": "200m", "memory": "256Mi"}
                        )
                        resp = appsV1Api.replace_namespaced_deployment(name=i, namespace=app_namespace, body=resp)
                        
        except ApiException as e:
            print(f"Exception when calling AppsV1Api->replace_namespaced_deployment: {e}")

        #get deployments replicas
        try:
            if len(run_list) > 0:
                for i in run_list:
                    resp = appsV1Api.read_namespaced_deployment(name=i, namespace=app_namespace)
                    print(f"{i} replicas: {resp.spec.replicas}")
        except ApiException as e:
            print(f"Exception when calling AppsV1Api->read_namespaced_deployment: {e}")


        #create HPAs
        try:
            if len(run_list) > 0:
                for i in run_list:
                    hpa_body = client.V2HorizontalPodAutoscaler()
                    hpa_body.metadata = client.V1ObjectMeta(name=f"{i}-hpa")
                    hpa_body.spec = client.V1HorizontalPodAutoscalerSpec(
                        max_replicas=MAX_REPLICAS,
                        min_replicas=MIN_REPLICAS,
                        scale_target_ref=client.V1CrossVersionObjectReference(
                            api_version="apps/v1",
                            kind="Deployment",
                            name=i
                        ),
                        target_cpu_utilization_percentage=80,                        
                    )
                    resp = autoscalingV1Api.create_namespaced_horizontal_pod_autoscaler(namespace=app_namespace, body=hpa_body )
            
        except ApiException as e:
            print(f"Exception when calling AutoscalingV1Api->create_namespaced_horizontal_pod_autoscaler: {e}")

        
        print(f"trying scaling every {ACTION_TIME} seconds")
        if len(run_list) > 0:
            while True:
                for i in run_list:
                    try:
                        num_replicas = random.randint(MIN_REPLICAS, MAX_REPLICAS)
                        print(f"Scaling {i} to {num_replicas} replicas")
                        resp = appsV1Api.patch_namespaced_deployment_scale(name=i, namespace=app_namespace, body={"spec": {"replicas": num_replicas}})
                    except:
                        print("Problem in scaling")
                time.sleep(ACTION_TIME)
            
    except ApiException as e:
        print("errore nelle call api")
        sys.exit(1)
    
    # print("Listing pods with their IPs:")
    # ret = coreV1Api.list_pod_for_all_namespaces()
    # for item in ret.items:
    #     print("%s\t%s\t%s" %
    #     (item.status.pod_ip,
    #         item.metadata.namespace,
    #         item.metadata.name))


    
    


    #list pods
    



if __name__ == "__main__":
    main()



def random_scaling():
    print("Start")