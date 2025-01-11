import os
import json

# Define the list of worker nodes and their corresponding GPU devices
worker_nodes_gpus = {
    "c315-001": ["GPU:0", "GPU:1", "GPU:2"],
    "c315-003": ["GPU:0", "GPU:1", "GPU:2"]
}

# Convert the worker_nodes_gpus dictionary to TF_CONFIG format
tf_config = {
    "cluster": {"worker": [f"{node}:{','.join(gpus)}" for node, gpus in worker_nodes_gpus.items()]},
    "task": {"type": "worker", "index": 0}
}

# Set TF_CONFIG environment variable
os.environ["TF_CONFIG"] = json.dumps(tf_config)

# Check if TF_CONFIG environment variable is set
tf_config_env = os.environ.get('TF_CONFIG')

if tf_config_env:
    # Parse TF_CONFIG JSON string
    tf_config_json = json.loads(tf_config_env)
    
    # Check if there are multiple worker nodes
    if "cluster" in tf_config_json and "worker" in tf_config_json["cluster"]:
        worker_nodes = tf_config_json["cluster"]["worker"]
        num_worker_nodes = len(worker_nodes)
        
        if num_worker_nodes > 1:
            print("Running on multiple nodes.")
            print("Node names and GPU devices:")
            for node in worker_nodes:
                print(node)
        else:
            print("Running on a single node.")
else:
    print("TF_CONFIG environment variable not found.")

