import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load data
data_dir = "data"
img_height = 250
img_width = 250
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

num_classes = len(train_ds.class_names)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential(
    [
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
        layers.RandomZoom(0.1),
        layers.RandomRotation(0.1),

        layers.Conv2D(16, 5, padding="same", activation="relu"),
        layers.Conv2D(64, 5, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Dropout(0.2),

        # layers.Conv2D(32, 3, padding="same", activation="relu"),
        # layers.Conv2D(64, 3, padding="same", activation="relu"),
        # layers.MaxPool2D(),
        # layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes)
    ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=30
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.save("models/new_bear.h5")

