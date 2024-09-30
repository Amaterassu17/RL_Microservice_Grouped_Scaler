import threading
import subprocess
import os
import signal
import time
import math
import random
import prometheus_server

class LocustClientThread(threading.Thread):
    def __init__(self, script_path, user_class, initial_users, max_users, spawn_rate, run_time=None, host=None, csv_prefix=None, log_file=None, load_type="random"):
        super().__init__()
        self.script_path = script_path
        self.user_class = user_class
        self.initial_users = initial_users
        self.max_users = max_users
        self.spawn_rate = spawn_rate
        self.run_time = run_time
        self.host = host
        self.csv_prefix = csv_prefix
        self.log_file = log_file
        self.load_type = load_type
        self.process = None

    def run(self):
        if self.load_type == "exponential":
            self.exponential_load()
        elif self.load_type == "gaussian":
            self.gaussian_load()
        elif self.load_type == "burst":
            self.burst_load()
        elif self.load_type == "random":
            self.random_load()
        elif self.load_type == "low":
            self.random_load(min_users=1, max_users=10)
        elif self.load_type == "medium":
            self.random_load(min_users=20, max_users=50)
        elif self.load_type == "high":
            self.random_load(min_users=100, max_users=200)
        else:
            raise ValueError("Invalid load type specified")

    def exponential_load(self):
        start_time = time.time()
        while not self._should_stop(start_time):
            current_time = time.time() - start_time
            num_users = self.initial_users * math.exp(self.spawn_rate * current_time)
            num_users = min(self.max_users, int(num_users))
            self.run_locust(num_users)
            time.sleep(1)
        self.stop_locust()

    def gaussian_load(self):
        mean = (self.initial_users + self.max_users) / 2
        std_dev = (self.max_users - self.initial_users) / 6
        start_time = time.time()
        while not self._should_stop(start_time):
            num_users = int(random.gauss(mean, std_dev))
            num_users = max(self.initial_users, min(self.max_users, num_users))
            self.run_locust(num_users)
            time.sleep(1)
        self.stop_locust()

    def burst_load(self):
        burst_interval = 60
        burst_duration = 10
        base_load = self.initial_users
        burst_load = self.max_users - self.initial_users
        start_time = time.time()
        while not self._should_stop(start_time):
            current_time = time.time() - start_time
            if int(current_time) % burst_interval < burst_duration:
                num_users = base_load + burst_load
            else:
                num_users = base_load
            self.run_locust(num_users)
            time.sleep(1)
        self.stop_locust()

    def random_load(self, min_users=None, max_users=None):
        if min_users is None:
            min_users = self.initial_users
        if max_users is None:
            max_users = self.max_users
        start_time = time.time()
        while not self._should_stop(start_time):
            num_users = random.randint(min_users, max_users)
            self.run_locust(num_users)
            time.sleep(1)
        self.stop_locust()

    def _should_stop(self, start_time):
        if self.run_time is not None:
            return time.time() - start_time >= self.run_time
        return False

    def run_locust(self, users):
        command = [
            "locust",
            "-f", self.script_path,
            "--headless",
            "-u", str(users),
            "-r", str(self.spawn_rate),
            "--class", self.user_class,
        ]
        if self.host:
            command.extend(["--host", self.host])
        if self.csv_prefix:
            command.extend(["--csv", self.csv_prefix])
        if self.log_file:
            command.extend(["--logfile", self.log_file])

        if self.process:
            os.kill(self.process.pid, signal.SIGTERM)
        self.process = subprocess.Popen(command)

    def stop(self):
        if self.process:
            os.kill(self.process.pid, signal.SIGTERM)
            self.process = None

    def stop_locust(self):
        if self.process:
            os.kill(self.process.pid, signal.SIGTERM)
            self.process = None

def start_locust(app_name, host ,initial_users, max_users, spawn_rate, run_time=None , csv_prefix=None, log_file=None, load_type="random"):
    script_path = f"./locustfile_{app_name}.py"
    user_class = "UserBehavior"
    # prometheus_server.start_http_server(8000)
    locust_thread = LocustClientThread(script_path, user_class, initial_users, max_users, spawn_rate, run_time, host, csv_prefix, log_file, load_type)
    locust_thread.start()
    return locust_thread

def stop_locust(locust_thread):
    locust_thread.stop()
