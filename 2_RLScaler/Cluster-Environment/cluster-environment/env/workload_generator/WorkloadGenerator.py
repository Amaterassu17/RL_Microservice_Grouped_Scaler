import random
import threading
import time
from .locust_client import LocustClientThread, start_locust, stop_locust
import os

# Assuming you have the previous code for LocustClientThread and start_locust, stop_locust

USER_CLASSES = [
    "LowLoadUser",
    "MediumLoadUser",
    "HighLoadUser"
    # "ExponentialLoadUser",
    # "GaussianLoadUser",
    # "RandomBurstsUser"
]

MIN_USERS = 1
MAX_USERS = 150
MIN_SPAWN_RATE = 1
MAX_SPAWN_RATE = 10
MIN_DURATION = 30  # minimum duration for each workload plan in seconds
MAX_DURATION = 300  # maximum duration for each workload plan in seconds

class RandomWorkloadGenerator():
    def __init__(self,app_name, host, log_dir) -> None:
        self.terminated = False
        self.current_thread = None
        self.dir_name = os.path.dirname(os.path.realpath(__file__))
        self.script_path = f"{self.dir_name}/locustfile_{app_name}.py"
        self.host = host
        self.log_dir = log_dir
        self.status= None


    def generate_random_workload(self):
        self.current_thread = None
        self.set_terminated(False)
        while not self.terminated:
            user_class = random.choice(USER_CLASSES)
            users = random.randint(MIN_USERS, MAX_USERS)
            spawn_rate = random.randint(MIN_SPAWN_RATE, MAX_SPAWN_RATE)
            duration = random.randint(MIN_DURATION, MAX_DURATION)

            # print(f"Starting workload: {user_class} with {users} users and spawn rate {spawn_rate} for {duration} seconds.")
            self.status = f"---> Current Workload: {user_class}, {users} users, spawn rate {spawn_rate} for {duration} seconds."
            
            # Start a new LocustClientThread
            locust_thread = start_locust(self.script_path, user_class, users, spawn_rate, self.host, duration, self.log_dir)
            locust_thread.join()
            # Stop the previous thread if it's still running
            if self.current_thread:
                stop_locust(self.current_thread)

            self.current_thread = locust_thread

            # Sleep for the duration before starting a new workload
            time.sleep(1)
        
        # Stop the last thread
        self.current_thread.join()

    def start_workload_simulation(self):
        workload_thread = threading.Thread(target=self.generate_random_workload)
        workload_thread.start()
        return workload_thread
    
    def set_terminated(self, terminated):
        self.terminated = terminated

    def get_status(self):
        return self.status



# # Usage
# script_path = "locustfile_teastore_old.py"
# host = "http://172.16.36.5:30080/tools.descartes.teastore.webui/"
# workload_generator = RandomWorkloadGenerator("teastore", host)
# workload_thread = workload_generator.start_workload_simulation(script_path, host)

# # Run your RL training episode here
# # ...
# print("initiating episode")
# time.sleep(10)
# workload_generator.set_terminated(True)
# print("ending episode")

# When the episode is done, kill the thread
