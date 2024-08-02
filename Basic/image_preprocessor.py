import os
import tensorflow as tf
import logging

# Setup logging
log_dir = 'C:/Projects/ObjectDetection/safety construction/logs'
log_file = os.path.join(log_dir, 'image_preprocessor.log')

# Define log format to include time
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handler = logging.FileHandler(log_file)
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger('ImagePreprocessorLogger')

class ImagePreprocessor:
    def __init__(self, target_size=(416, 416), normalize=True):
        self.target_size = target_size
        self.should_normalize = normalize  # Renamed to avoid conflict
        logger.info(f'ImagePreprocessor initialized with target size: {target_size} and normalize set to {normalize}')

    def resize(self, image):
        try:
            image = tf.image.resize(image, self.target_size)
            logger.info('Image resized successfully.')
            return image
        except Exception as e:
            logger.error(f'Error resizing image: {e}')
            raise

    def normalize(self, image):
        try:
            image = image / 255.0
            logger.info('Image normalized successfully.')
            return image
        except Exception as e:
            logger.error(f'Error normalizing image: {e}')
            raise

    def preprocess(self, image):
        try:
            image = self.resize(image)
            if self.should_normalize:  # Use renamed attribute
                image = self.normalize(image)
            logger.info('Image preprocessing completed.')
            return image
        except Exception as e:
            logger.error(f'Error in preprocessing image: {e}')
            raise

    def augment(self, image):
        try:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
            logger.info('Image augmentation completed.')
            return image
        except Exception as e:
            logger.error(f'Error in augmenting image: {e}')
            raise

    def process_folder(self, folder_path, output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, filename)
                    output_image_path = os.path.join(output_path, filename)
                    
                    image = tf.io.read_file(image_path)
                    image = tf.image.decode_image(image, channels=3)
                    processed_image = self.preprocess(image)
                    processed_image = self.augment(processed_image)
                    
                    # Save processed image
                    tf.keras.preprocessing.image.save_img(output_image_path, processed_image)
                    logger.info(f'Processed and saved image: {output_image_path}')
        
        except Exception as e:
            logger.error(f'Error processing folder {folder_path}: {e}')
            raise

def main():
    dataset_dir = 'C:/Projects/ObjectDetection/safety construction/dataset'
    output_dir = 'C:/Projects/ObjectDetection/safety construction/processed_dataset'
    
    # Initialize the ImagePreprocessor
    preprocessor = ImagePreprocessor()

    # Define folders
    folders = ['train', 'valid', 'test']

    for folder in folders:
        input_folder = os.path.join(dataset_dir, folder, 'images')
        output_folder = os.path.join(output_dir, folder, 'images')
        
        # Process images in each folder
        preprocessor.process_folder(input_folder, output_folder)
    
if __name__ == "__main__":
    main()

