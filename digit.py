import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
# The dataset contains 60,000 training images and 10,000 test images
# Each image is 28x28 grayscale, and the labels are digits from 0-9
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Dataset loaded successfully.")
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Preprocess the data
# 1. Normalize image pixel values from [0, 255] to [0, 1]
#    This helps with model convergence and performance
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Convert labels to one-hot encoded vectors
#    For example, digit '5' becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#    This is required for categorical cross-entropy loss
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(f"One-hot encoded training labels shape: {y_train.shape}")
print(f"One-hot encoded test labels shape: {y_test.shape}")

# Build the MLP model
# A Sequential model is a linear stack of layers.
# Flatten: Converts the 2D image (28x28) into a 1D vector (784 features).
# Dense: Standard fully connected neural network layer.
# Dropout: Randomly sets a fraction of input units to 0 at each update during training time,
#          which helps prevent overfitting.
print("Building MLP model...")
model = Sequential([
    Flatten(input_shape=(28, 28)), # Input layer: Flattens 28x28 images to 784-dimensional vectors
    Dense(512, activation='relu'),  # Hidden layer 1 with 512 neurons and ReLU activation
    Dropout(0.2),                   # Dropout layer to prevent overfitting
    Dense(256, activation='relu'),  # Hidden layer 2 with 256 neurons and ReLU activation
    Dropout(0.2),                   # Another dropout layer
    Dense(num_classes, activation='softmax') # Output layer with 10 neurons (one for each digit)
                                            # Softmax activation for multi-class classification
])

# Compile the model
# Optimizer: Adam is a good default optimizer for many tasks.
# Loss function: Categorical Cross-entropy is used for multi-class classification with one-hot encoded labels.
# Metrics: Accuracy is used to monitor the performance during training and evaluation.
print("Compiling model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Train the model
# epochs: Number of times to iterate over the entire training dataset.
# batch_size: Number of samples per gradient update.
# validation_split: Fraction of the training data to be used as validation data.
#                   The model will not be trained on this data and will evaluate the loss
#                   and any model metrics on this data at the end of each epoch.
print("Training model...")
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.1,
                    verbose=1) # verbose=1 shows a progress bar

print("Model training complete.")

# Evaluate the model on the test data
# This gives an unbiased estimate of the model's performance on unseen data.
print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize training loss and accuracy
print("Generating plots for training history...")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()
print("Plots displayed.")

# Optional: Make a prediction on a sample image
print("\nMaking a prediction on a sample test image...")
sample_image = x_test[0] # Get the first image from the test set
true_label = np.argmax(y_test[0]) # Get its true label

# Add a batch dimension to the sample image for prediction (model expects a batch)
sample_image_reshaped = np.expand_dims(sample_image, axis=0)

# Predict the probabilities for each class
predictions = model.predict(sample_image_reshaped)
predicted_label = np.argmax(predictions[0]) # Get the class with the highest probability

print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")

# Display the sample image
plt.figure(figsize=(2, 2))
plt.imshow(sample_image * 255, cmap='gray') # Multiply by 255 to display correctly
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.axis('off')
plt.show()
