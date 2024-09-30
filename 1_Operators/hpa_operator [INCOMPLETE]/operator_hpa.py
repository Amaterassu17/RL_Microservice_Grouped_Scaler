import random
import kopf
from kubernetes import client, config, watch

import time
import os
import threading
from util import *
import signal
import shutil
import math
from prom_crawler_new import PromCrawlerNew


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
def startup_fn(memo: kopf.Memo, **kwargs):
    print("Starting up...")

    if str(os.getenv('KUBERNETES_POD')).lower().strip() == "true":
        print("using kube pod config")
        config.load_incluster_config()  # works well in a pod, No bearer token please}")
    else:
        print("Using kubeconfig")
        config.load_kube_config()

    # get prometheus address bu asking the address of the first node through kubernetes
    client_v1 = client.CoreV1Api()
    nodes = client_v1.list_node()
    first_node = nodes.items[0].metadata.name
    print(f"First node: {first_node}")
    prom_address = f"http://{first_node}:{PROMETHEUS_PORT}"

    prom_crawler = PromCrawlerNew(prom_address)


    config_operator = OperatorConfigs()

    if os.getenv('APP_NAME') is not None:
        config_operator.app_name = os.getenv('APP_NAME')

    if os.getenv('NAMESPACE') is not None:
        config_operator.namespace = os.getenv('NAMESPACE')

    if os.getenv('GROUP_FILE') is not None:
        print("Group file is set")
        config_operator.group_file = os.getenv('GROUP_FILE')
        config_operator.read_groups(os.getenv('GROUP_FILE'))
    else:
        print("Group file is not set")
        os.kill(os.getpid(), signal.SIGTERM)

    if os.getenv('NUM_NODES') is not None:
        print(f"Number of nodes: {os.getenv('NUM_NODES')}")
        config_operator.number_of_nodes = int(os.getenv('NUM_NODES'))

    memo["config_operator"] = config_operator
    memo["barrier"] = threading.Barrier(len(config_operator.groups))
    memo["barrier_ext"] = threading.Barrier(len(config_operator.groups))
    memo["prom_crawler"] = prom_crawler
    print("Config Initialized")


@kopf.on.delete('apps', 'v1', 'deployments')
def app_delete_handler(memo: kopf.Memo, spec, **kwargs):
    config_operator = memo["config_operator"]
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
            resp = config_operator.appsV1.patch_namespaced_deployment(name=run, namespace=config_operator.namespace,
                                                                      body={"metadata": {"finalizers": []}})
            print(f"Deleted finalizers for {run}")

        except Exception as e:
            print(f"Problem in deleting finalizers: {e}")
    print("Cleaned up")
    return {'message': 'cleaned up'}


@kopf.daemon('apps', 'v1', 'deployments')
def daemon_sync_scaling(memo: kopf.Memo, stopped, **kwargs):
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

            # memo["barrier_ext"].wait()
            print(f"Starting Group HPA for {selected_group}")
            while not stopped:
                # stopped.wait(ACTION_INTERVAL)  ##let the microservices be up for ACTION_INTERVAL seconds, so that every Node is running
                # time.sleep(ACTION_STOP_INTERVAL)
                
                num_replicas = compute_hpa_replicas_group(selected_group, config_operator, memo["prom_crawler"])

                print(
                    f"Calling function with num_replicas: {num_replicas}, selected_group: {selected_group}, count_of_tries: {config_operator.count_of_tries + 1}")
                scaling_schema_group(config_operator, num_replicas, selected_group,
                                        memo["experiment"][selected_group][config_operator.count_of_tries][
                                            num_replicas], memo["barrier"])
                time.sleep(2)


                memo["barrier_ext"].wait()

                # if config_operator.count_of_tries >= config_operator.number_of_tries:
                #     test_finished = True

            # print memo experiment selected group to display everything inside
            # export_scaling_times(memo["experiment"][selected_group], selected_group, config_operator)

            # time.sleep(5)

            # os.kill(os.getpid(), signal.SIGTERM)


        else:
            print("Killing daemon")
    else:
        print("Not the desired app")


def scaling_schema_group(config_operator, num_replicas, group, timestamp_dict, barrier):
    try:
        running_for_group = 0
        print(f"Scaling {config_operator.groups[group]} to {num_replicas} replicas")
        print(config_operator.groups[group])

        microservices_in_group = config_operator.groups[group]

        # iterative
        try:
            resp = config_operator.appsV1.patch_namespaced_deployment_scale(
            name=microservices_in_group, namespace=config_operator.namespace,
            body={"spec": {"replicas": num_replicas}}
            )
            running_for_group += len(microservices_in_group)
        except Exception as e:
            print(f"Problem in scaling: {e}")
            os.kill(os.getpid(), signal.SIGTERM)

        groups_copy = config_operator.groups[group][:]
        while len(groups_copy) > 0 and running_for_group > 0:
            time.sleep(TIME_PRECISION)
            for microservice in groups_copy:
                if microservice in config_operator.run_states.keys():
                    try:
                        resp = config_operator.appsV1.read_namespaced_deployment_status(name=microservice,
                                                                                        namespace=config_operator.namespace)
                        ready_replicas = resp.status.ready_replicas

                        pod_list = config_operator.coreV1.list_namespaced_pod(namespace=config_operator.namespace,
                                                                              label_selector=f"run={microservice}")
                        # Count the number of pods that are in 'Running' state and ready
                        ready_pods = [pod for pod in pod_list.items if pod.status.phase == 'Running' and
                                      any(cond.type == 'Ready' and cond.status == 'True' for cond in
                                          pod.status.conditions)]

                        # Count the number of pods in the 'Terminating' state
                        terminating_pods = [pod for pod in pod_list.items if
                                            pod.metadata.deletion_timestamp is not None]

                        if ready_replicas == num_replicas and len(ready_pods) == num_replicas and len(
                                terminating_pods) == 0:
                            # print(f"Deployment {microservice} is ready")
                            timestamp_dict[microservice].time_last_scaled_ready = time.time()
                            timestamp_dict[microservice].time_total = timestamp_dict[
                                                                          microservice].time_last_scaled_ready - \
                                                                      timestamp_dict[
                                                                          microservice].time_before_api
                            groups_copy.remove(microservice)
                            print(f"Time taken for {microservice}: {timestamp_dict[microservice].time_total}")
                            break
                    except Exception as e:
                        print(f"Problem in reading replicas: {e}")
                        os.kill(os.getpid(), signal.SIGTERM)

        barrier.wait()

        # reset all the replicas in the group to 1
        for microservice in config_operator.groups[group]:
            if microservice in config_operator.run_states.keys():
                try:
                    resp = config_operator.appsV1.patch_namespaced_deployment_scale(
                        name=microservice, namespace=config_operator.namespace,
                        body={"spec": {"replicas": 1}}
                    )
                    # print(f"Reset {microservice} to 1 replica")
                except Exception as e:

                    print(f"Problem in rescaling: {e} to 1 replica")
                    os.kill(os.getpid(), signal.SIGTERM)

        groups_copy = config_operator.groups[group][:]
        while len(groups_copy) > 0 and running_for_group > 0:
            time.sleep(TIME_PRECISION)
            for microservice in groups_copy:
                if microservice in config_operator.run_states.keys():
                    try:
                        # Count the number of replicas that are assigned and are ready from the deployment
                        resp = config_operator.appsV1.read_namespaced_deployment_status(name=microservice,
                                                                                        namespace=config_operator.namespace)
                        pod_list = config_operator.coreV1.list_namespaced_pod(namespace=config_operator.namespace,
                                                                              label_selector=f"run={microservice}")

                        # Count the number of pods that are in 'Running' state and ready
                        ready_pods = [pod for pod in pod_list.items if pod.status.phase == 'Running' and
                                      any(cond.type == 'Ready' and cond.status == 'True' for cond in
                                          pod.status.conditions)]

                        # Count the number of pods in the 'Terminating' state
                        terminating_pods = [pod for pod in pod_list.items if
                                            pod.metadata.deletion_timestamp is not None]

                        ready_replicas = resp.status.ready_replicas
                        replicas = resp.spec.replicas

                        if int(ready_replicas) == 1 and int(resp.spec.replicas) == 1 and len(ready_pods) == 1 and len(
                                terminating_pods) == 0:
                            print(f"Deployment {microservice} is resetted")
                            groups_copy.remove(microservice)
                            break
                    except Exception as e:
                        print(f"Problem in reading replicas: {e}")
                        os.kill(os.getpid(), signal.SIGTERM)

        barrier.wait()
    except Exception as e:
        print(f"Problem in scaling: {e}")
        os.kill(os.getpid(), signal.SIGTERM)



def compute_hpa_replicas_group(selected_group, config_operator, prom_crawler):
    # get the metrics from the prometheus
    metric_cpu = metric_usage_cpu_hpa_formula.format(app_microservice=config_operator.groups[selected_group],
                                                     control_node=config_operator.control_node,
                                                     cpu_rate_window=config_operator.cpu_rate_window)

# experiment is made like this
# Dictionary with count_tries entries. Each entry has as value
# a dictionary with the microservices as keys and the scaling times as values
def export_scaling_times(experiment_dict, selected_group, config_operator):
    # for a group we create a file for the corresponding experiment
    ex_filename = "group" + str(selected_group) + "times.csv"
    ex_dirname = f"{config_operator.group_file.split('.')[0]}_nodes{config_operator.number_of_nodes}_tries{config_operator.number_of_tries}"
    try:
        if os.path.isdir(ex_dirname) and not os.path.islink(ex_dirname):
            shutil.rmtree(ex_dirname)
        elif os.path.exists(ex_dirname):
            os.remove(ex_dirname)

        os.makedirs(ex_dirname)

        with open(f"{ex_dirname}/{ex_filename}", "w") as f:

            if os.stat(f"{ex_dirname}/{ex_filename}").st_size == 0:
                f.write("Microservice | Replicas,")
                f.write(f"AvgTime, MinTime, MaxTime")
                f.write("\n")

            # average, min and max of all the total times from the various try for each microservice for the config_operator.number_of_tries

            # if it does exist append after a \n, otherwise create the file
            # format is microservice, time_before_api, time_after_api, time_last_scaled_ready, time_total
            # if created for the first time write the header
            # print(experiment_dict)
            for microservice in config_operator.groups[selected_group]:
                print(f"Considering {microservice}")
                for num_replicas in config_operator.group_desired_replicas[selected_group]:
                    print(f"Writing data for microservice: {microservice}, replicas: {num_replicas}")
                    f.write(f"{microservice}| R{num_replicas},")
                    total_times = []
                    for i in range(config_operator.number_of_tries):
                        total_times.append(experiment_dict[i][num_replicas][microservice].time_total)
                        # print(total_times)
                        # print(experiment_dict[i][num_replicas][microservice].time_total)

                    print(f"Total times for {microservice} with {num_replicas} replicas: {total_times}")
                    f.write(f"{sum(total_times) / len(total_times)},{min(total_times)},{max(total_times)}\n")

    except Exception as e:

        print(f"Problem in exporting scaling times: {e}")
        os.kill(os.getpid(), signal.SIGTERM)
