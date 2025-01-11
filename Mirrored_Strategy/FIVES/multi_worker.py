import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Function to load and preprocess the MNIST dataset
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    return model

# Function to define and compile the model within the distributed strategy scope
def define_compile_model(strategy):
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    return model

# Function to extract worker nodes from SLURM environment variables
def extract_worker_nodes():
    node_list = os.environ.get('SLURM_NODELIST')
    nodes = node_list.split(',')
    return nodes

# Read TF_CONFIG environment variable
tf_config_env = os.environ.get('TF_CONFIG')

if tf_config_env:
    # Parse TF_CONFIG JSON string
    tf_config_json = json.loads(tf_config_env)
    
    # Check if there are multiple worker nodes
    if "cluster" in tf_config_json and "worker" in tf_config_json["cluster"]:
        worker_nodes = tf_config_json["cluster"]["worker"]
        num_workers = len(worker_nodes)
        print("Running on multiple nodes with {} workers.".format(num_workers))
        
        # Initialize MultiWorkerMirroredStrategy
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        print("Running on a single node.")
        # Initialize MirroredStrategy for single-node training
        strategy = tf.distribute.MirroredStrategy()
else:
    print("TF_CONFIG environment variable not found. Running with default strategy.")
    # Initialize default strategy for single-node training
    strategy = tf.distribute.MirroredStrategy()

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

# Define and compile the model within the strategy scope
model = define_compile_model(strategy)

# Fit the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

