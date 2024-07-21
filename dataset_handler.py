import os
import tensorflow as tf
import logging
from logging.handlers import TimedRotatingFileHandler

# Setup logging
log_dir = r'C:\Study\Computer Vision\tensorflow\Tutorial\Tensorflow\logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'data_loader.log')

# Define log format to include time
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handler = TimedRotatingFileHandler(log_file, when='D', interval=1, backupCount=3)
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger('DataLoaderLogger')

class DataLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        logger.info(f'DataLoader initialized with dataset directory: {dataset_dir}')

    def load_data(self, subset):
        images_path = os.path.join(self.dataset_dir, subset, 'images')
        labels_path = os.path.join(self.dataset_dir, subset, 'labels')

        image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')])
        label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')])

        if len(image_files) == 0 or len(label_files) == 0:
            logger.error(f'No files found in {subset} dataset.')
            raise FileNotFoundError(f'No files found in {subset} dataset.')

        logger.info(f'Loaded {len(image_files)} images and {len(label_files)} labels from {subset} dataset.')
        return image_files, label_files

    def parse_function(self, image_path, label_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [416, 416]) / 255.0  # Resize to YOLO input size and normalize

            label = tf.io.read_file(label_path)
            # Add label processing logic here
            logger.info(f'Successfully parsed {image_path} and {label_path}')
            return image, label
        except Exception as e:
            logger.error(f'Error parsing {image_path} or {label_path}: {e}')
            raise

    def get_dataset(self, subset):
        try:
            image_files, label_files = self.load_data(subset)
            dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
            dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # Adjust batch size as needed
            logger.info(f'{subset} dataset prepared and batched.')
            return dataset
        except Exception as e:
            logger.error(f'Error preparing {subset} dataset: {e}')
            raise
    
    def count_labels(self, subset):
        images_path = os.path.join(self.dataset_dir, subset, 'images')
        labels_path = os.path.join(self.dataset_dir, subset, 'labels')

        label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')])

        if len(label_files) == 0:
            logger.error(f'No label files found in {subset} dataset.')
            raise FileNotFoundError(f'No label files found in {subset} dataset.')

        logger.info(f'Counted {len(label_files)} labels from {subset} dataset.')
        return len(label_files)

def main():
    # Define the path to your dataset
    dataset_dir = r'C:/Projects/ObjectDetection/safety construction/dataset'
    
    # Initialize the DataLoader
    data_loader = DataLoader(dataset_dir)
    
    # Load datasets
    train_dataset = data_loader.get_dataset('train')
    valid_dataset = data_loader.get_dataset('valid')
    test_dataset = data_loader.get_dataset('test')
    
    # Count images in each dataset
    train_image_count = count_images(train_dataset)
    valid_image_count = count_images(valid_dataset)
    test_image_count = count_images(test_dataset)
    
    # Count labels in each dataset
    train_label_count = data_loader.count_labels('train')
    valid_label_count = data_loader.count_labels('valid')
    test_label_count = data_loader.count_labels('test')
    
    # Print the results
    print(f"Total number of images in train_dataset: {train_image_count}")
    print(f"Total number of images in valid_dataset: {valid_image_count}")
    print(f"Total number of images in test_dataset: {test_image_count}")
    
    print(f"Total number of labels in train_dataset: {train_label_count}")
    print(f"Total number of labels in valid_dataset: {valid_label_count}")
    print(f"Total number of labels in test_dataset: {test_label_count}")

def count_images(dataset):
    num_images = 0
    for images, _ in dataset:
        num_images += images.shape[0]
    return num_images

if __name__ == "__main__":
    main()

    def parse_function(self, image_path, label_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [416, 416]) / 255.0  # Resize to YOLO input size and normalize

            label = tf.io.read_file(label_path)
            # Add label processing logic here
            logger.info(f'Successfully parsed {image_path} and {label_path}')
            return image, label
        except Exception as e:
            logger.error(f'Error parsing {image_path} or {label_path}: {e}')
            raise

    def get_dataset(self, subset):
        try:
            image_files, label_files = self.load_data(subset)
            dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
            dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # Adjust batch size as needed
            logger.info(f'{subset} dataset prepared and batched.')
            return dataset
        except Exception as e:
            logger.error(f'Error preparing {subset} dataset: {e}')
            raise
    
    def count_labels(self, subset):
        images_path = os.path.join(self.dataset_dir, subset, 'images')
        labels_path = os.path.join(self.dataset_dir, subset, 'labels')

        label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')])

        if len(label_files) == 0:
            logger.error(f'No label files found in {subset} dataset.')
            raise FileNotFoundError(f'No label files found in {subset} dataset.')

        logger.info(f'Counted {len(label_files)} labels from {subset} dataset.')
        return len(label_files)

def main():
    # Define the path to your dataset
    dataset_dir = 'C:/Projects/ObjectDetection/safety construction/dataset'
    
    # Initialize the DataLoader
    data_loader = DataHandler(dataset_dir)
    
    try:
        # Load datasets
        train_dataset = data_loader.get_dataset('train')
        valid_dataset = data_loader.get_dataset('valid')
        test_dataset = data_loader.get_dataset('test')
    
        # Count images in each dataset
        train_image_count = count_images(train_dataset)
        valid_image_count = count_images(valid_dataset)
        test_image_count = count_images(test_dataset)
    
        # Count labels in each dataset
        train_label_count = data_loader.count_labels('train')
        valid_label_count = data_loader.count_labels('valid')
        test_label_count = data_loader.count_labels('test')
    
        # Print the results
        print(f"Total number of images in train_dataset: {train_image_count}")
        print(f"Total number of images in valid_dataset: {valid_image_count}")
        print(f"Total number of images in test_dataset: {test_image_count}")
    
        print(f"Total number of labels in train_dataset: {train_label_count}")
        print(f"Total number of labels in valid_dataset: {valid_label_count}")
        print(f"Total number of labels in test_dataset: {test_label_count}")
    except Exception as e:
        logger.error(f'Error in main: {e}')

def count_images(dataset):
    num_images = 0
    for images, _ in dataset:
        num_images += images.shape[0]
    return num_images

if __name__ == "__main__":
    main()
