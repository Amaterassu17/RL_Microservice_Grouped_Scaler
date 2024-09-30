import threading
import subprocess
import signal
import os
import time
from kubernetes import client, config

class LocustClientThread(threading.Thread):
    def __init__(self, script_path, user_class, users, spawn_rate, host, runtime, log_dir):
        super().__init__()
        self.script_path = script_path
        self.user_class = user_class
        self.users = users
        self.spawn_rate = spawn_rate
        self.process = None
        self.host= host
        self.runtime = runtime
        self.log_dir = log_dir

    def run(self):
        
        command = f"locust -f '{self.script_path}' --headless -u {self.users} -r {self.spawn_rate} --class {self.user_class} --host '{self.host}' --run-time {self.runtime} --skip-log-setup"

        # if self.log_dir:
        #     command += f" --logfile {self.log_dir}/locust.log"
        
        with open(f"{self.log_dir}/locust.log", "w") as f:
            f.write("")
            self.process = subprocess.Popen(command, shell=True, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
            )
            time.sleep(self.runtime)
            self.stop()
        

    def stop(self):
        if self.process:
            os.kill(self.process.pid, signal.SIGTERM)
            self.process = None

def start_locust(script_path, user_class, users, spawn_rate, host=None, runtime=30, log_dir= None):
    # print(f"Running command: locust -f {script_path} --headless -u {users} -r {spawn_rate} --class {user_class} --host {host}")
    locust_thread = LocustClientThread(script_path, user_class, users, spawn_rate,host, runtime, log_dir)
    locust_thread.start()
    return locust_thread

def stop_locust(locust_thread):
    locust_thread.stop()
