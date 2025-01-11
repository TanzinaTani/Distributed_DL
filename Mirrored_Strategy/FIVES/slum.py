import os
import subprocess
import socket

def print_slurm_nodes_and_hostname():
    node_list = os.getenv('SLURM_JOB_NODELIST')
    if node_list is None:
        print("SLURM_JOB_NODELIST is not set. Make sure this script is running in a Slurm managed job.")
        return

    result = subprocess.run(['scontrol', 'show', 'hostnames', node_list], check=True, stdout=subprocess.PIPE)
    nodes = result.stdout.decode('utf-8').strip().split()
    hostname = socket.gethostname()

    print("Slurm nodes:", nodes)
    print("Current hostname:", hostname)

print_slurm_nodes_and_hostname()

