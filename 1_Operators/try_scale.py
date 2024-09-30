import random
from kubernetes import client, config
import sys
from kubernetes.client import configuration
import yaml
import argparse
import asyncio
import time
import os
from pick import pick

MIN_REPLICAS=1
MAX_REPLICAS=6
ACTION_TIME=120


def main():
    print("Start")

    parser = argparse.ArgumentParser()
    parser.add_argument('--app_name', type=str, default='teastore')
    parser.add_argument('--app_namespace', type=str, default='default')
    parser.add_argument('--groups', type=int, default=1)

    #add other things

    options = parser.parse_args()
    app_name = options.app_name
    app_namespace = options.app_namespace
    groups = options.groups


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
        AutoscalerV2Api = client.V2HorizontalPodAutoscaler()
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
        



        if groups == run_list.len():
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
        else:
            print(f"Scaling in {groups} groups")
            print(f"Scaling every {ACTION_TIME} seconds")
            print("Groups")
            #partition 
            groups_list = divide_run_list_random_groups(run_list, groups) 
            print(groups_list)
            if len(run_list) > 0:
                while True:
                    for i in groups_list:
                        num_replicas = random.randint(MIN_REPLICAS, MAX_REPLICAS)
                        print(f"Scaling group {i} to {num_replicas} replicas")

                        for j in i:
                            try:
                                resp = appsV1Api.patch_namespaced_deployment_scale(name=j, namespace=app_namespace, body={"spec": {"replicas": num_replicas}})
                            except:
                                print("Problem in group scaling")
                    time.sleep(ACTION_TIME)


            
    except:
        print("call error ")
        sys.exit(1)
    
    # print("Listing pods with their IPs:")
    # ret = coreV1Api.list_pod_for_all_namespaces()
    # for item in ret.items:
    #     print("%s\t%s\t%s" %
    #     (item.status.pod_ip,
    #         item.metadata.namespace,
    #         item.metadata.name))


    
    


    #list pods
    
def divide_run_list_random_groups(run_list, num_groups):
    random.shuffle(run_list)
    return [run_list[i::num_groups] for i in range(num_groups)]

if __name__ == "__main__":
    main()


