import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import os


class SSD:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = applications.VGG16(include_top=False, input_shape=self.input_shape)
        base_output = base_model.output

        # Adding SSD layers
        conv4_3_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1), name='conv4_3_norm')(base_output)
        x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv6')(conv4_3_norm)
        x = layers.Conv2D(1024, (1, 1), padding='same', activation='relu', name='conv7')(x)

        # Additional SSD layers for different scales
        # conv8_1, conv8_2
        conv8_1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu', name='conv8_1')(x)
        conv8_2 = layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu', name='conv8_2')(conv8_1)
        
        # conv9_1, conv9_2
        conv9_1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu', name='conv9_1')(conv8_2)
        conv9_2 = layers.Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation='relu', name='conv9_2')(conv9_1)

        # Class prediction and bounding box regression layers
        # For simplicity, assuming fixed number of boxes per feature map location
        num_boxes = 4
        class_preds = []
        box_preds = []
        for feature_map in [conv4_3_norm, x, conv8_2, conv9_2]:
            class_pred = layers.Conv2D(num_boxes * (self.num_classes + 1), (3, 3), padding='same')(feature_map)
            class_pred = layers.Reshape((-1, self.num_classes + 1))(class_pred)
            class_preds.append(class_pred)

            box_pred = layers.Conv2D(num_boxes * 4, (3, 3), padding='same')(feature_map)
            box_pred = layers.Reshape((-1, 4))(box_pred)
            box_preds.append(box_pred)

        class_preds = layers.Concatenate(axis=1)(class_preds)
        box_preds = layers.Concatenate(axis=1)(box_preds)

        model = models.Model(inputs=base_model.input, outputs=[class_preds, box_preds])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', 
                           loss={'conv2d_22': 'categorical_crossentropy', 
                                 'conv2d_23': 'mean_squared_error'})

def load_dataset(image_dir, label_dir, input_shape):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]

    images = []
    labels = []

    for img_file, lbl_file in zip(image_files, label_files):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_shape[:2]) / 255.0

        with open(lbl_file, 'r') as file:
            label = np.array([list(map(float, line.split())) for line in file])

        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def train_ssd_model(model, train_images, train_labels, epochs=10, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for step, (images, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                class_predictions, box_predictions = model(images, training=True)
                class_loss = tf.keras.losses.categorical_crossentropy(labels, class_predictions)
                box_loss = tf.keras.losses.mean_squared_error(labels, box_predictions)
                loss = class_loss + box_loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if step % 10 == 0:
                print(f'Step {step}, Loss: {loss.numpy()}')

    return model

def main():
    input_shape = (300, 300, 3)  # SSD typically uses 300x300 input size
    num_classes = 20  # Update with your number of classes

    # Initialize SSD model
    ssd_model = SSD(input_shape, num_classes)
    ssd_model.compile_model()
    ssd_model.model.summary()

    # Load dataset
    image_dir = 'path/to/images'
    label_dir = 'path/to/labels'
    train_images, train_labels = load_dataset(image_dir, label_dir, input_shape)

    # Train model
    trained_model = train_ssd_model(ssd_model.model, train_images, train_labels, epochs=10, batch_size=32)

    # Save model
    trained_model.save('ssd_custom.h5')

if __name__"__main__":
    main()
