from __future__ import absolute_import, division, print_function, unicode_literals
import os as os
import tensorflow as tf
tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Download and inspect the dataset
import pathlib
data_root = pathlib.Path("/home/paul/Pictures")
print(data_root)
for item in data_root.iterdir():
    print(item)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

#Inspect the images

#Determine the label for each image

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

#Assign an index to each label:

label_to_index = dict((name, index) for index,name in enumerate(label_names))

#Create a list of every file, and its label index

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

#Load and format the images

img_path = all_image_paths[0]

img_raw = tf.read_file(img_path)

#Decode it into an image tensor:

img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
#print(img_tensor.dtype)

#Resize it for your model:

img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
#print(img_final.numpy().min())
#print(img_final.numpy().max())

#Wrap up these up in simple functions for later

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [28, 28])
  image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

#In [0]:

#import matplotlib.pyplot as plt

#image_path = all_image_paths[0]
#label = all_image_labels[0]

#plt.imshow(load_and_preprocess_image(img_path))
#plt.grid(False)
#plt.xlabel(caption_image(img_path).encode('utf-8'))
#plt.title(label_names[label].title())
#print()

#Build a tf.data.Dataset

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

#The output_shapes and output_types fields describe the content of each item in the dataset. In this case it is a set of scalar binary-strings

#print('shape: ', repr(path_ds.output_shapes))
#print('type: ', path_ds.output_types)
#print()
#print(path_ds)

#Now create a new dataset that loads and formats images on the fly by mapping preprocess_image over the dataset of paths

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


#import matplotlib.pyplot as plt

#plt.figure(figsize=(8,8))
#for n,image in enumerate(image_ds.take(4)):
 # plt.subplot(2,2,n+1)
 # plt.imshow(image)
 # plt.grid(False)
 # plt.xticks([])
 # plt.yticks([])
 # plt.xlabel(caption_image(all_image_paths[n]))
 # plt.show()

#A dataset of (image, label) pairs


label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


#for label in label_ds.take(10):
 # print(label_names[label.numpy()])

#Since the datasets are in the same order we can just zip them together to get a dataset of (image, label) pairs.


image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


#print(image_label_ds)

#Note: When you have arrays like all_image_labels and all_image_paths an alternative to tf.data.dataset.Dataset.zip is to slice the pair of arrays.

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

#Basic methods for training

BATCH_SIZE = 8

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)

#There are a few things to note here:


ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)



#code for cnn
from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2D(12, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(12, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(12, (3, 3), activation='relu'))
model.add(layers.Flatten())
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
#checkpoint_path = "/home/paul/computervisionproject/cnn/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
#    period=2)
#model.save_weights(checkpoint_path.format(epoch=0))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds, epochs=6, steps_per_epoch=35)# callbacks = [cp_callback])
img_final=tf.expand_dims(load_and_preprocess_image("/home/paul/banana.jpg"), axis=0)
prediction = model.predict(img_final)
for i, logits in enumerate(prediction):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = label_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
 #   print("Example {} prediction: {} )".format(i, name,))



    #import time
#saved_model_path = "./saved_models/{}".format(int(time.time()))

#tf.keras.experimental.export_saved_model(model, saved_model_path)
#saved_model_path
