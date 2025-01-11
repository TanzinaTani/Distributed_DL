import os
import json
import subprocess

# Function to get individual node names from SLURM_NODELIST
def get_slurm_node_info():
    # Retrieve the SLURM_NODELIST and SLURMD_NODENAME environment variables
    slurm_nodelist = os.getenv('SLURM_NODELIST')
    slurmd_nodename = os.getenv('SLURMD_NODENAME')

    node_list = []

    if slurmd_nodename:
        print("Current SLURM Node Name:", slurmd_nodename)
    else:
        print("SLURMD_NODENAME environment variable is not set.")

    if slurm_nodelist:
        try:
            # Use scontrol to expand SLURM_NODELIST into individual node names
            result = subprocess.check_output(['scontrol', 'show', 'hostnames', slurm_nodelist])
            node_list = result.decode('utf-8').strip().split('\n')
            print("Individual SLURM Node List:", node_list)
        except subprocess.CalledProcessError as e:
            print("Could not expand SLURM_NODELIST due to an error:", e)
    else:
        print("SLURM_NODELIST environment variable is not set.")

    return node_list, slurmd_nodename

# Getting node info
node_list, current_node = get_slurm_node_info()

# Assuming each node name can be used as a hostname
worker_hosts = [f"{node}:12345" for node in node_list]

# Finding the index of the current node in the list to set the correct 'task' index
current_node_index = node_list.index(current_node) if current_node in node_list else 0

# Setting the TF_CONFIG environment variable
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': worker_hosts
    },
    'task': {'type': 'worker', 'index': current_node_index}
})

print("TF_CONFIG set to:", os.environ["TF_CONFIG"])

