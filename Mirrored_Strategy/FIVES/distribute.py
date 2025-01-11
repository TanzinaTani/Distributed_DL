import tensorflow as tf
import os
import json


import tensorflow as tf
import os
import json

train_image_folder = '/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/dataset/FIVES_dataset/train/Original'
train_mask_folder = '/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/dataset/FIVES_dataset/train/Ground_truth'

valid_image_folder = '/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/dataset/FIVES_dataset/validation/Original'
valid_mask_folder = '/scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/dataset/FIVES_dataset/validation/Ground_truth'

#Use os.listdir() to get the list of file names in each folder
train_image_files = os.listdir(train_image_folder)
train_mask_files = os.listdir(train_mask_folder)

valid_image_files = os.listdir(valid_image_folder)
valid_mask_files = os.listdir(valid_mask_folder)

#Create full file paths
train_image_paths = [os.path.join(train_image_folder, filename) for filename in train_image_files]
train_mask_paths = [os.path.join(train_mask_folder, filename) for filename in train_mask_files]

valid_image_paths = [os.path.join(valid_image_folder, filename) for filename in valid_image_files]
valid_mask_paths = [os.path.join(valid_mask_folder, filename) for filename in valid_mask_files]

train_image_paths = sorted(train_image_paths)
train_mask_paths = sorted(train_mask_paths)

valid_image_paths = sorted(valid_image_paths)



# Set up TF_CONFIG environment variable
worker_nodes_gpus = {
    "c315-001": ["GPU:0", "GPU:1", "GPU:2"],
    "c315-003": ["GPU:0", "GPU:1", "GPU:2"]
}

ps_nodes = ["c315-001:12345", "c315-003:12345"] 

# Convert the worker_nodes_gpus dictionary to TF_CONFIG format
tf_config = {
    "cluster": {"worker": [f"{node}:{','.join(gpus)}" for node, gpus in worker_nodes_gpus.items()],
     "ps": ps_nodes},
    "task": {"type": "worker", "index": 0}
}

# Set TF_CONFIG environment variable
os.environ["TF_CONFIG"] = json.dumps(tf_config)

BATCH_SIZE = 8
IMAGE_SIZE = 256

# Define the data processing functions
def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1] for mask images
        image = tf.where(image >= 0.5, 1.0, 0.0)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1] for RGB images
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    return image

def augment_data(image, mask):
    # Randomly flip the image and mask horizontally
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Randomly flip the image and mask vertically
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Randomly rotate the image and mask in range -30 to 30 degrees
    angle = tf.random.uniform((), minval=-30, maxval=30, dtype=tf.float32)
    image = tf.image.rot90(image, k=tf.cast(angle / 30, dtype=tf.int32))
    mask = tf.image.rot90(mask, k=tf.cast(angle / 30, dtype=tf.int32))

    return image, mask

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

# Define the model creation function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define the number of workers and the worker index
num_workers = len(worker_nodes_gpus)
worker_index = 0  # Set to 0 for simplicity, but you may need to adjust this

# Define the sharding and batching parameters
shuffle_buffer_size_train = len(train_image_paths)
shuffle_buffer_size_val = len(valid_image_paths)

# Create the dataset creation and augmentation function within the distributed strategy scope
def create_distributed_datasets(image_list, mask_list, augment=False, batch_size=None, shuffle_buffer_size=None):
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))

    # Shard the dataset based on the number of workers and the worker index
    dataset = dataset.shard(num_shards=num_workers, index=worker_index)

    # Shuffle the dataset if shuffle_buffer_size is provided
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)

    # Load and preprocess the data
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Augment the data if augment is True
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset if batch_size is provided
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the dataset for performance improvement
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create the distributed train and validation datasets
train_dataset = create_distributed_datasets(train_image_paths, train_mask_paths, augment=True, batch_size=BATCH_SIZE, shuffle_buffer_size=shuffle_buffer_size_train)
val_dataset = create_distributed_datasets(valid_image_paths, valid_mask_paths, batch_size=BATCH_SIZE, shuffle_buffer_size=shuffle_buffer_size_val)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

# Define the distribution strategy
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
strategy = tf.distribute.experimental.ParameterServerStrategy(resolver)

# Create the model within the distributed strategy scope
with strategy.scope():
    model = create_model()

# Compile and train your model using the distributed datasets and strategy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

