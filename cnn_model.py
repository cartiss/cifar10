import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# Prepare training dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = np.array(X_train).astype('float') / 255.
X_test = np.array(X_test).astype('float') / 255.

X_val = X_train[:10000]
X_train = X_train[10000:]
y_val = y_train[:10000]
y_train = y_train[10000:]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Main model
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='selu', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10)
])

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Fit train data
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Evaluate test data
model.evaluate(X_test, y_test)


# Plot loss and accuracy
pd.DataFrame(history.history).plot(figsize=(15, 8))
plt.grid(True)
plt.gca()
plt.show()

# Save model
model.save("models/cnn_cifar10.h5")
