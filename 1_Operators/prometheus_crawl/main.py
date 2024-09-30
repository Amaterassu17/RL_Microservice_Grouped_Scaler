import os
import argparse
import json

from prometheus_adaptor import *

PROM_URL='http://localhost:9090' # default Prometheus URL but should be changed if used NodePort
PROM_TOKEN = None # can be changed but default config has none



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_name', type=str)  # name of the metric to query
    # parser.add_argument('--labels', type=str, default='')  # labels for the metric query
    parser.add_argument('--window', type=int, default=3600)  # time window for the metric query
    parser.add_argument('--step', type=str, default='15s')  # time step for the metric query
    parser.add_argument('--timeout', type=int, default=5*60)  # how much time does the script run for
    parser.add_argument('--interval', type=int, default=5)  # how often the script calls prometheus API (in seconds)
    parser.add_argument('--prom_host', type=str, default='http://localhost:9090')  # Prometheus address
    parser.add_argument('--prom_token', type=str, default=None)  # Prometheus token
    options = parser.parse_args()

    print(options.step)

    if not options.metric_name:
        print('Please specify the metric name!')
        exit()

    if(options.window):
        print('Querying metric', options.metric_name, 
            #   'with labels', options.labels,
                'in the last', options.window, 'seconds')
    else:
        print('Querying metric', options.metric_name, 
              'with labels', options.labels)

    # set up the prometheus client
    global PROM_URL
    global PROM_TOKEN
    if os.getenv('PROM_HOST'):
        PROM_URL = os.getenv('PROM_HOST')
        print("Prometheus address: ", PROM_URL)
    elif options.prom_host:
        PROM_URL = options.prom_host
        print("Prometheus address: ", PROM_URL)

    if os.getenv('PROM_TOKEN'):
        PROM_TOKEN = os.getenv('PROM_TOKEN')
        print("Prometheus token: ", PROM_TOKEN)
    elif options.prom_token:
        PROM_TOKEN = options.prom_token
        print("Prometheus token: ", PROM_TOKEN)


    prometheus_adaptor = PromCrawler(prom_address=PROM_URL, prom_token=PROM_TOKEN)

    
    prometheus_adaptor.update_period(crawling_period=options.window)
    prometheus_adaptor.update_step(step=options.step)

    for i in range(0, options.timeout, options.interval):
        print('\n\n')
        print('Time:', prometheus_adaptor.get_current_time())

        myquery = options.metric_name
            
        # data = prometheus_adaptor.fetch_data_range(my_query=options.metric_name)
        data = prometheus_adaptor.fetch_data_range_in_chunks(my_query=myquery)
        traces={}
    
        # pretty_json = json.dumps(data, indent=4)
        pretty_json = json.dumps(data, indent=4)
        # print(data)

        traces= prometheus_adaptor.get_promdata(myquery,traces,"retrieved_metric", "php-apache")
        # print(traces)
        exit(0)
        time.sleep(options.interval)
        



    

    


if __name__ == '__main__':
    main()