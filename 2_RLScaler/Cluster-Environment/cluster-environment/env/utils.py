# imports

#ENVIRONMENT CONSTANTS
TIME_TERMINATED = 600 #SHOULD REFLECT FOR HOW MUCH WE WANT TO TRAIN THE SCENARIO
#KEEP IN MIND THE LENGTH OF WORKLOADS OR SCENARIOS THAT YOU WOULD CONSIDER AN EPISODE



CPU_LIMIT = 1024
MEMORY_LIMIT = 1024
MIN_REPLICAS = 1
MAX_REPLICAS = 10

MAX_MOCK_LATENCY = 10000.0

TIME_API_CHECK_REPLICAS = 0.1


#LOGGING CHECKPOINTING

CHECKPOINT_DIR = './checkpoints/'
LOG_DIR = './logs/'
PLOT_FIG = True
SAVE_FIG = True
SAVE_TO_FILE = True
DATA_PATH = 'data.csv'



#HYPER PARAMETERS

TOTAL_ITERATIONS = 2
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


REWARD_AVG_THRESHOLD = 100
REWARD_STD_THRESHOLD = 10

ILLEGAL_PENALTY = -1


#REWARD PARAMETERS

CPU_IMPORTANCE = 0.8
MEM_IMPORTANCE = 0.2


#METRICS Precisions

NUM_TRIES_CLUSTER_STABLE = 8



#Range of time in seconds to which the metrics are fetched
METRICS_RANGE = 1 # 2 minutes
PROMETHEUS_STEP = 3 # 3 seconds

TIME_STABILIZATION = 10
STDDEV_TIME_WINDOW = 30 # seconds
AVG_TIME_WINDOW = 30 # seconds
CPU_RATE_WINDOW = 30
STDDEV_CPU_THRESHOLD= 0.1
STDDEV_MEM_THRESHOLD= 100000000
latency_percentile = 0.95

metric_stddev_cpu = """
stddev_over_time(
  (
    (sum by (container) (rate(container_cpu_usage_seconds_total{{container=~"{app_microservice}",node!="{control_node}"}}[{cpu_rate_window}s:1s])))
    /
    ignoring(container) group_left 
    sum(kube_node_status_capacity{{resource="cpu",node!="{control_node}"}})
  )[{stddev_window}s:1s]
)
"""

metric_stddev_mem = """
stddev_over_time(
  (
    (sum by (container) (container_memory_usage_bytes{{container=~"{app_microservice}",node!="{control_node}"}}))
    /
    ignoring(container) group_left 
    sum(kube_node_status_capacity{{resource="memory",node!="{control_node}"}})
  )[{stddev_window}s:1s]
)
"""

metric_cpu = """
avg_over_time(
  (
    (sum by (container) (rate(container_cpu_usage_seconds_total{{container=~"{app_microservice}",node!="{control_node}"}}[{cpu_rate_window}s:1s])))
    /
    ignoring(container) group_left 
    sum(kube_node_status_capacity{{resource="cpu",node!="{control_node}"}})
  )[{avg_window}s:1s]
)
"""

metric_mem = """
avg_over_time(
  (
    (sum by (container) (container_memory_usage_bytes{{container=~"{app_microservice}",node!="{control_node}"}}))
    /
    ignoring(container) group_left 
    sum(kube_node_status_capacity{{resource="memory",node!="{control_node}"}})
  )[{avg_window}s:1s]
)
"""

metric_latency = """
avg_over_time(histogram_quantile({latency_percentile}, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_service=~"{app_microservice}"}}[{latency_rate_window}s:1s])) by (le))[3s:1s])
"""

metric_usage_cpu_hpa_formula = """
avg_over_time(
  ( avg(
    sum by (container,pod) (
      rate(container_cpu_usage_seconds_total{{container=~"{app_microservice}", node!="{control_node}"}}[{cpu_rate_window}s:1s])
    )
    /
    sum by (container,pod) (
      kube_pod_container_resource_limits{{container=~"{app_microservice}", resource="cpu", node!="{control_node}"}}
    )) by (container)
  )[60s:3s]
)
"""

metric_usage_mem_hpa_formula = """
avg_over_time(
  ( avg(
    sum by (container,pod) (
      container_memory_usage_bytes{{container=~"{app_microservice}", node!="{control_node}"}}
    )
    /
    sum by (container,pod) (
      kube_pod_container_resource_limits{{container=~"{app_microservice}", resource="memory", node!="{control_node}"}}
    )) by (container)
  )[60s:3s]
)
"""


metric_replicas = """
kube_deployment_status_replicas_ready{{deployment=~"{app_microservice}"}}
"""


# Classes

class SingletonMeta(type):
    """
    Singleton metaclass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
