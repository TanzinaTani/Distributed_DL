import json 
#from  DeepLabv3_256  import *
#from load_data import *
#from test_data import *
import keras
from keras import backend as K

import tensorflow as tf

from datetime import datetime
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '

#os.environ["TF_CONFIG"] = json.dumps({"cluster": {"worker": ["c315-001:12345", "c315-003:12345"]}, "task": {"type": "worker", "index": 0}})

import os
import json

# Define the list of worker nodes and their corresponding GPU devices
worker_nodes_gpus = {
    "c301-001": ["GPU:0", "GPU:1", "GPU:2"],
    "c301-003": ["GPU:0", "GPU:1", "GPU:2"]
    # 'c302-001': ['GPU:0', 'GPU:1', 'GPU:2'], 'c302-002': ['GPU:0', 'GPU:1', 'GPU:2']
}
#c302-001
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




# Function to print available devices based on the strategy
def print_available_devices(strategy):
    worker_devices = strategy.extended.worker_devices
    for device in worker_devices:
        print("Available device:", device)

# Define MirroredStrategy with specific devices
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])

#strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Print available devices
print_available_devices(strategy)

#options = tf.data.Options()

# Set auto sharding policy to DATA
#options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

from  DeepLabv3_256  import *
from load_data import *
from test_data import *

# Apply the options object to the dataset
train_dataset = train_dataset.with_options(options)

val_dataset = val_dataset.with_options(options)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

#with strategy.scope():
    #model = create_model()
    #model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = bce_dice_loss, metrics=[dice_coef, "accuracy"])
  #  return model

global_batch_size = 8  # Example, adjust based on your training configuration

with strategy.scope():
    # Adjusted to capture global_batch_size, if necessary
    def build_and_compile_cnn_model():
        model = create_model()
        # Pass a lambda or wrapper function that includes global_batch_size if needed

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = bce_dice_loss, metrics=[dice_coef, "accuracy"])
        return model
    
    multi_worker_model = build_and_compile_cnn_model()



import time


class TotalTimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        self.total_training_time = time.time() - self.train_start_time
        print(f"Total training time: {self.total_training_time:.2f} seconds")


time_history = TotalTimeHistory()

# Define the checkpoint directory to store the checkpoints.
#checkpoint_dir = '/scratch/09825/dtu14/project/final_project/transformer/training_checkpoints'

#checkpoints = ModelCheckpoint(
#    checkpoint_dir,
#    monitor= 'val_loss',
#    verbose=0,
#    save_best_only=True,
#    save_weights_only=False,
#    mode='auto',
#)

EPOCHS = 100


history = multi_worker_model.fit(
    train_dataset,
    validation_data=val_dataset,
    callbacks=[time_history],
    epochs=EPOCHS
)

#plt.plot(history.history["loss"])
#plt.title("Training Loss")
#plt.ylabel("loss")
plt.xlabel("epoch")
#plt.savefig("/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/Training_ms_five_loss.png")

plt.figure()  
plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
#plt.savefig("/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/Training_ms_five_accuracy.png")

plt.figure()  
plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
#plt.savefig("/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/Val_ms_five_loss.png")

plt.figure()  
plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
#plt.savefig("/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/Val_ms_five_accuracy.png")


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection + 1e-7) / (union + 1e-7)

def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(np.maximum(y_true, y_pred))
    return (intersection + 1e-7) / (union + 1e-7)

# Assuming your mask values are binary (0 and 1)
def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true.flatten(), y_pred.flatten())

# Other metrics can be computed using Scikit-learn functions
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true.flatten(), y_pred.flatten())
    recall = recall_score(y_true.flatten(), y_pred.flatten())
    f1 = f1_score(y_true.flatten(), y_pred.flatten())
    return acc, precision, recall, f1


import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef

def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true.numpy().flatten(), y_pred.numpy().flatten())

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true.numpy().flatten(), y_pred.numpy().flatten())
    precision = precision_score(y_true.numpy().flatten(), y_pred.numpy().flatten(), zero_division=1)
    recall = recall_score(y_true.numpy().flatten(), y_pred.numpy().flatten(), zero_division=1)
    f1 = f1_score(y_true.numpy().flatten(), y_pred.numpy().flatten(), zero_division=1)
    return acc, precision, recall, f1

def calculate_metrics(test_dataset, model ):
    dice_scores = []
    iou_scores = []
    mcc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    threshold=0.5

    for image, true_mask in test_dataset:
        pred_mask = model.predict(image)
        pred_mask = tf.cast(pred_mask > threshold, dtype=tf.int32)
        true_mask = tf.cast(true_mask > threshold, dtype=tf.int32)

        dice = dice_coefficient(true_mask.numpy(), pred_mask.numpy())
        iou_score = iou(true_mask.numpy(), pred_mask.numpy())
        mcc_score = mcc(true_mask, pred_mask)
        accuracy, precision, recall, f1 = compute_metrics(true_mask, pred_mask)

        dice_scores.append(dice)
        iou_scores.append(iou_score)
        mcc_scores.append(mcc_score)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_mcc = np.mean(mcc_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)

    print("Mean Dice coefficient:", mean_dice)
    print("Mean IoU (Jaccard) score:", mean_iou)
    print("Mean Matthews Correlation Coefficient:", mean_mcc)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean Precision:", mean_precision)
    print("Mean Recall (Sensitivity):", mean_recall)
    print("Mean F1 score:", mean_f1)


calculate_metrics(test_dataset,  multi_worker_model)
#calculate_metrics(test_dataset1,  multi_worker_model)
