import os
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
from prometheus_api_client.utils import parse_datetime

from util import *
from concurrent.futures import ThreadPoolExecutor
import json


class PromCrawlerNew:
    prom_address = None
    crawling_period = None
    step = '15s'
    chunk_sz = 900 #? What is this?

    def __init__(self, prom_address=None):
        self.prom_address = prom_address or os.getenv("PROM_HOST")

        self.prom = PrometheusConnect(url=self.prom_address, disable_ssl=True)

        print("Prometheus address: ", self.prom_address)
        
        if not self.prom_address:
            raise ValueError(
                "Please appropriately configure environment variables $PROM_HOST, $PROM_TOKEN, $CRAWLING_PERIOD to successfully run the crawler and profiler!")


    def get_current_time(self):
        current_time_str = datetime.fromtimestamp(self.now).strftime("%I:%M:%S")
        return current_time_str
    
    def fetch_metric_range(self, metric, start, end, step):
        try:
            result = self.prom.custom_query_range(
                query=metric,
                start_time=start,
                end_time=end,
                step=step
            )
            return result
        except Exception as e:
            print(e)
            return None
        
    def fetch_metric(self, metric):
        #without range
        try:
            result = self.prom.custom_query(
                query=metric
            )
            return result
        except Exception as e:
            print(e)
            return None

    def fetch_metrics(self, metrics, step = 0):

       
        results = {}

        if step != 0:
            end = datetime.now()
            start = end - timedelta(seconds=self.crawling_period)
            step = PROMETHEUS_STEP
            # metrics is a dict of metric names and their queries

            for metric in metrics.keys():
                try:
                    result = self.fetch_metric_range(metrics[metric], start, end, step)
                    results[metric] = result
                except Exception as e:
                    print(e)
                    results[metric] = None

        else:
            for metric in metrics.keys():
                try:
                    result = self.fetch_metric(metrics[metric])
                    results[metric] = result
                    
                except Exception as e:
                    print(e)
                    results[metric] = None

        return results
   
    


# Other utility functions for the PromCrawler 
def get_key_name(attribute, klist):
    keys = [kname for kname in klist if attribute in kname.lower()]
    if len(keys) > 0:
        return keys[0]
    else:
        return ""
