"""
Handwritten Digit Recognition using CNN
Developed by Sujal Sariya
Part of the RISE Internship Program (Tamizhan Skills)
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

print("🔁 Loading and preprocessing data...")

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize & reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

print("🧠 Building CNN model...")

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Training the model...")
model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Predict & visualize
predictions = model.predict(x_test)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {predictions[i].argmax()}, Label: {y_test[i]}")
    plt.axis('off')
    plt.show()
