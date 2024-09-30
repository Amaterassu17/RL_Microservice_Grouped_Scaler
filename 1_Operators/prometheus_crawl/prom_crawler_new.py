from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import os


class PromCrawlerNew:
    now = None
    start = None
    crawling_period = 3600
    step = '15s'
    chunk_sz = 900

    def __init__(self, prom_address=None, prom_token=None):
        self.prom_address = prom_address or os.getenv("PROM_HOST")
        self.prom_token = prom_token or os.getenv("PROM_TOKEN")

        self.prom = PrometheusConnect(url=self.prom_address, headers={"Authorization": f"Bearer {self.prom_token}"},
                                      disable_ssl=True)

        print("Prometheus address: ", self.prom_address)
        print("Prometheus token: ", self.prom_token)
        if not self.prom_address or not self.crawling_period:
            raise ValueError(
                "Please appropriately configure environment variables $PROM_HOST, $PROM_TOKEN, $CRAWLING_PERIOD to successfully run the crawler and profiler!")

    def update_period(self, crawling_period):
        self.crawling_period = crawling_period
        self.now = datetime.now()
        self.start = self.now - timedelta(seconds=self.crawling_period)
        self.end = self.now

    def fetch_data_range(self, my_query, start, end):
        try:
            result = self.prom.custom_query_range(
                query=my_query,
                start_time=start,
                end_time=end,
                step=self.step
            )
            return result
        except Exception as e:
            print(e)
            return None

    def fetch_data_range_in_chunks(self, my_query):
        all_metric_history = []
        cur_start = self.start

        while cur_start < self.end:
            cur_end = min(cur_start + timedelta(seconds=self.chunk_sz), self.end)

            trials = 0
            cur_metric_history = None
            while cur_metric_history is None and trials < 3:
                cur_metric_history = self.fetch_data_range(my_query, cur_start, cur_end)
                trials += 1

            if cur_metric_history is None:
                continue

            all_metric_history.extend(cur_metric_history)
            cur_start = cur_end

        return all_metric_history

    def get_promdata(self, query, traces, resourcetype):
        cur_trace = self.fetch_data_range_in_chunks(query)

        if not cur_trace:
            print(f"There are no data points for metric query {query}.")
            return traces

        metric_obj_attributes = cur_trace[0]["metric"].keys()
        pod_key_name = get_key_name("pod", metric_obj_attributes)
        container_key_name = get_key_name("container", metric_obj_attributes)
        ns_key_name = get_key_name("namespace", metric_obj_attributes)
        if ns_key_name == "":
            ns_key_name = get_key_name("ns", metric_obj_attributes)

        if pod_key_name == "" or container_key_name == "" or ns_key_name == "":
            print(
                f"[Warning] The metric object returned from Prometheus query {query} does not have required attribute tags.")
            print(f"[Warning] - pod attribute name: {pod_key_name}")
            print(f"[Warning] - container attribute name: {container_key_name}")
            print(f"[Warning] - namespace attribute name: {ns_key_name}")

        for metric_obj in cur_trace:
            try:
                pod = metric_obj["metric"][pod_key_name]
            except:
                continue

            try:
                container = metric_obj["metric"][container_key_name]
                if container == "POD":
                    continue
            except:
                continue

            metrics = metric_obj['values']
            traces = construct_nested_dict(traces, container, resourcetype, pod)
            traces[container][resourcetype][pod].extend(metrics)
        return traces


# Other utility functions for the PromCrawler class
def construct_nested_dict(traces_dict, container, resourcetype, pod=None):
    if pod is None:
        if container not in traces_dict.keys():
            traces_dict[container] = {resourcetype: []}
        elif resourcetype not in traces_dict[container].keys():
            traces_dict[container][resourcetype] = []
    else:
        if container not in traces_dict.keys():
            traces_dict[container] = {resourcetype: {pod: []}}
        elif resourcetype not in traces_dict[container].keys():
            traces_dict[container][resourcetype] = {pod: []}
        elif pod not in traces_dict[container][resourcetype].keys():
            traces_dict[container][resourcetype][pod] = []

    return traces_dict


def get_key_name(attribute, klist):
    keys = [kname for kname in klist if attribute in kname.lower()]
    if len(keys) > 0:
        return keys[0]
    else:
        return ""
