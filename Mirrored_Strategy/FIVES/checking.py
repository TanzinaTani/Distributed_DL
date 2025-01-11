import os
import json
import tensorflow as tf

# Function to set TF_CONFIG based on Slurm environment
def set_tf_config_from_slurm():
    import subprocess
    import socket
    
    node_list = os.getenv('SLURM_JOB_NODELIST')
    if node_list is None:
        raise EnvironmentError("SLURM_JOB_NODELIST is not set. Make sure this is running in a Slurm environment.")
    
    result = subprocess.run(['scontrol', 'show', 'hostnames', node_list], check=True, stdout=subprocess.PIPE)
    nodes = result.stdout.decode('utf-8').strip().split()
    nodename = socket.gethostname()
    
    if nodename in nodes:
        task_index = nodes.index(nodename)
    else:
        raise ValueError("Current node is not in the SLURM_JOB_NODELIST")
    
    tf_config = {
        "cluster": {"worker": [f"{node}:12345" for node in nodes]},
        "task": {"type": "worker", "index": task_index}
    }
    
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print("TF_CONFIG set on node:", nodename)

# Set the TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Set to '2' to see fewer messages

# Initialize TF_CONFIG
set_tf_config_from_slurm()

# Define TensorFlow strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Function to print available devices based on the strategy
def print_available_devices(strategy):
    worker_devices = strategy.extended.worker_devices
    for device in worker_devices:
        print("Available device:", device)

# Print available devices
print_available_devices(strategy)

