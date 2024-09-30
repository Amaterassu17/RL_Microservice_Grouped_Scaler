from prom_crawler_new import PromCrawlerNew
from utils import *
import json
import time
import numpy as np
from kubernetes import client, config

app_name = "teastore"
app_namespace = "default"
control_node = "chiclet-6.lille.grid5000.fr"
namespace = "default"
microservices = ["teastore-webui", "teastore-db", "teastore-persistence", "teastore-auth", "teastore-recommender", "teastore-image"]

def main():
    print("Promtheus Test")

    #kubeconfig and kube variables
    config.load_kube_config()
    prom_address = None
    kube_client = client.ApiClient()
    coreV1 = client.CoreV1Api()
    appsV1 = client.AppsV1Api()
    control_node_name = None
    control_node_IP = None
    #check if kube is working
    try:
        nodes = coreV1.list_node()
        resp = coreV1.list_namespaced_pod(namespace=app_namespace)

        #TAKE NODE ADDRESS AND PORT (SHOULD BE 9100)
        if nodes.items:
            first_node = nodes.items[0]
            for address in first_node.status.addresses:
                if address.type == "InternalIP":
                    print(f"Node Name: {first_node.metadata.name}, IP Address: {address.address}")
                    control_node_name = first_node.metadata.name
                    control_node_IP = address.address
                    break

            if prom_address != None:
                prom_address = prom_address
            else:
                prom_address = f"http://{first_node.status.addresses[0].address}:30090"
    except Exception as e:
        print(e)
        raise Exception("Kubernetes is not working")

    prom = PromCrawlerNew(prom_address=prom_address)



    container_filter = f"{app_name}-.*"
    node_filter = f"{control_node}"

    avg_cpu_util_microservice = {}
    avg_mem_util_microservice = {}
    stability_index_cpu_microservice = {}
    stability_index_mem_microservice = {}
    num_replicas = {}

    metrics = {
            'metricCPU': metric_cpu.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW),
            'metricMEM': metric_mem.format(app_microservice=container_filter, control_node=node_filter, avg_window=AVG_TIME_WINDOW),
            'metricSTDCPU': metric_stddev_cpu.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_CPU_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW),
            'metricSTDMEM': metric_stddev_mem.format(app_microservice=container_filter, control_node=node_filter, threshold=STDDEV_MEM_THRESHOLD, stddev_window=STDDEV_TIME_WINDOW),
            # latency metric TBD
            # 'metricLATENCY': metric_latency.format(app_microservice=app_name, namespace=namespace)
        }

    for num_tries in range(1, NUM_TRIES_CLUSTER_STABLE + 1):
        fetched_metrics = prom.fetch_metrics(metrics, step= 3)
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

   

    
    for microservice in microservices:
        try:
            resp = appsV1.read_namespaced_deployment(name=microservice, namespace=app_namespace)
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

    print(json.dumps(dict, indent=3)
    )

    #try some rewards







    #test 
    # microservices_queries = {}

    # microservices_queries["metric_CPU"] = metric_cpu.format(app_microservice=f"{app_name}-.+", control_node=control_node)
    # microservices_queries["metric_MEM"] = metric_mem.format(app_microservice=f"{app_name}-.+", control_node=control_node)
    # microservices_queries["metric_STDDEV_CPU"] = metric_stddev_cpu.format(app_microservice=f"{app_name}-.+", control_node=control_node)
    # microservices_queries["metric_STDDEV_MEM"] = metric_stddev_mem.format(app_microservice=f"{app_name}-.+", control_node=control_node)

  

    # # for microservice in microservices:
    # #     metric_CPU = metric_cpu.format(app_microservice=microservice, control_node=control_node)
    # #     microservices_queries[microservice] = metric_CPU
    
    # # # for microservice in microservices:
    # # #     self.get_metric_CPU(app_name=microservice, control_node=control_node)
    # # #     self.get_metric_MEM(app_name=microservice, control_node=control_node)
    # # #     self.get_metric_STDDEV_CPU(app_name=microservice, control_node=control_node)
    # # #     self.get_metric_STDDEV_MEM(app_name=microservice, control_node=control_node)
    # #     # self.get_metric_LATENCY(app_name=microservice, namespace=namespace)

    # # for microservice in microservices:
    # #     print(microservice)
    # #     print(microservices_queries[microservice])
    # #     result = self.fetch_metric(microservices_queries[microservice])
    # #     result=json.dumps(result, indent=3)
    # #     print(result)
    
    # # print(metric_CPU)
    # # result = prom.fetch_metric(metric_CPU)
    # # result=json.dumps(result, indent=3)

    # results = prom.fetch_metrics(microservices_queries, step=0)
    # results=json.dumps(results, indent=3)
    # print(results)

if  __name__ == "__main__":
    main()