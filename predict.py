import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

img_height = 250
img_width = 250

model = tf.keras.models.load_model("models/bears_94.h5")

image_name = input("Filename? ")

img = tf.keras.utils.load_img(
    "inputs/" + image_name, target_size=(250, 250)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ["Black Bear", "Grizzly Bear", "Panda", "Polar Bear", "Teddy Bear"]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)