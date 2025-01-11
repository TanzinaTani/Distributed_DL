import os

# valid_mask_paths

import os
import tensorflow as tf
BATCH_SIZE = 16
IMAGE_SIZE = 256

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

    # Randomly rotate the image and mask in range -30 to 30 degree
    angle = tf.random.uniform((), minval=-30, maxval=30, dtype=tf.float32)
    image = tf.image.rot90(image, k=tf.cast(angle / 30, dtype=tf.int32))
    mask = tf.image.rot90(mask, k=tf.cast(angle / 30, dtype=tf.int32))

    return image, mask


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list, augment=False, batch_size=None, shuffle_buffer_size=None):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)


    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

# Split the dataset based on the Horovod rank
def get_dataset_for_rank(train_image_files, train_mask_files, valid_image_files, valid_mask_files, rank, size):

    total_samples = len(train_image_files)
    samples_per_rank = total_samples // size
    start = rank * samples_per_rank
    end = start + samples_per_rank if rank != size - 1 else total_samples

    total_samples = len(valid_image_files)
    samples_per_rank = total_samples // size
    start_label = rank * samples_per_rank
    end_label = start_label + samples_per_rank if rank != size - 1 else total_samples

    return train_image_files[start:end],train_mask_files[start:end], valid_image_files[start_label:end_label],valid_mask_files[start_label:end_label]



def get_dataset(rank, size):


    #Define the paths to the dataset folders
    train_image_folder = 'dataset/MSD_split/train/images'
    train_mask_folder = 'dataset/MSD_split/train/mask'

    valid_image_folder = 'dataset/MSD_split/valid/images'
    valid_mask_folder = 'dataset/MSD_split/valid/mask'


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
    valid_mask_paths = sorted(valid_mask_paths)


    #Define the paths to the dataset folders
    train_image_folder1 = 'dataset/FIVES_dataset/train/Original'
    train_mask_folder1 = 'dataset/FIVES_dataset/train/Ground_truth'

    valid_image_folder1 = 'dataset/FIVES_dataset/validation/Original'
    valid_mask_folder1 = 'dataset/FIVES_dataset/validation/Ground_truth'


    #Use os.listdir() to get the list of file names in each folder
    train_image_files1 = os.listdir(train_image_folder1)
    train_mask_files1 = os.listdir(train_mask_folder1)

    valid_image_files1 = os.listdir(valid_image_folder1)
    valid_mask_files1 = os.listdir(valid_mask_folder1)

    #Create full file paths
    train_image_paths1 = [os.path.join(train_image_folder1, filename) for filename in train_image_files1]
    train_mask_paths1 = [os.path.join(train_mask_folder1, filename) for filename in train_mask_files1]

    valid_image_paths1 = [os.path.join(valid_image_folder1, filename) for filename in valid_image_files1]
    valid_mask_paths1 = [os.path.join(valid_mask_folder1, filename) for filename in valid_mask_files1]

    train_image_paths1 = sorted(train_image_paths1)
    train_mask_paths1 = sorted(train_mask_paths1)

    valid_image_paths1 = sorted(valid_image_paths1)
    valid_mask_paths1 = sorted(valid_mask_paths1)

    train_image_path = train_image_paths1 + train_image_paths
    valid_image_path = valid_image_paths1 + valid_image_paths

    train_mask_path = train_mask_paths1 + train_mask_paths
    valid_mask_path = valid_mask_paths1 + valid_mask_paths
    print("total len:", len(train_image_path))

    total_samples = len(train_image_path)
    samples_per_rank = total_samples // size
    start = rank * samples_per_rank
    end = start + samples_per_rank if rank != size - 1 else total_samples

    train_image_path = train_image_path[start:end]
    train_mask_path = train_mask_path[start:end]

    total_samples = len(valid_mask_path)
    samples_per_rank = total_samples // size
    start = rank * samples_per_rank
    end = start + samples_per_rank if rank != size - 1 else total_samples

    valid_image_path = valid_image_path[start:end]
    valid_mask_path = valid_mask_path[start:end]

    shuffle_buffer_size_train = len(train_image_path)
    shuffle_buffer_size_val = len(valid_image_path)
    print( "rank:", rank, " len train:", len(train_image_path), " len valid", len(valid_image_path))

    # Create augmented train_dataset with a specific batch size
    train_dataset = data_generator(train_image_path, train_mask_path, augment=True, batch_size=BATCH_SIZE, shuffle_buffer_size=shuffle_buffer_size_train)

    # Create validation dataset without augmentation
    val_dataset = data_generator(valid_image_path, valid_mask_path, batch_size=BATCH_SIZE, shuffle_buffer_size=shuffle_buffer_size_val)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset
