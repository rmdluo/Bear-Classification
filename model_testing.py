import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf

# from tensorflow import keras

model_name = input("Model name? ")

model = tf.keras.models.load_model("models/" + model_name)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_ds = tf.keras.utils.image_dataset_from_directory(
  "data",
  image_size=(250, 250),
  batch_size=32
)

loss, acc = model.evaluate(test_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))