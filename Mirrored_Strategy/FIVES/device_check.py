import tensorflow as tf

# List all logical devices
all_devices = tf.config.list_logical_devices()
gpu_devices = [device for device in all_devices if device.device_type == 'GPU']

if not gpu_devices:
    print("No GPU devices found.")
else:
    print("Found GPU devices:")
    for device in gpu_devices:
        print(f"Device name: {device.name}")
        print(f"Memory: {device.memory} MB")
        print(f"Device type: {device.device_type}")

