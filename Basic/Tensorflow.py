import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Initialize the Sequential model
model = Sequential()

# Add a Conv2D layer
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(64, 64, 3)
))

# Add a MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another Conv2D layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Add a Dense layer
model.add(Dense(units=128, activation='relu'))

# Add the output layer
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

