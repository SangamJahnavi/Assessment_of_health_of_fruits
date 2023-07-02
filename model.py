import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import pickle
dataset_url = "dataset\\train"
img_height = 100
img_width = 100
batch_size = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_url,
    validation_split=0.2,
    subset='training',
    seed=256,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_url,
    validation_split=0.2,
    subset="validation",
    seed=256,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
# print(class_names)
plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(
    buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(
    buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),
])
# test_url = "dataset\\test"

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     test_url,
#     seed=256,
#     image_size=(img_height, img_width),
#     shuffle=False  # No shuffling for classification report
# )
# test_images, test_labels = tuple(zip(*test_ds))

# predictions = model.predict(test_ds)
# score = tf.nn.softmax(predictions)
# results = model.evaluate(test_ds)
# print("Test loss, test acc:", results)
# y_test = np.concatenate(test_labels)
# y_pred = np.array([np.argmax(s) for s in score])

# print(classification_report(y_test, y_pred, target_names=class_names))

model = Sequential([
    # resacle to be between 0 and 1
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    data_augmentation,
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs = 16
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
pickle.dump(model, open('model.pkl', 'wb'))
modelp = pickle.load(open('model.pkl', 'rb'))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training accuracy')
# plt.plot(epochs_range, val_acc, label='Validation accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and validation accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training loss')
# plt.plot(epochs_range, val_loss, label='Validation loss')
# plt.legend(loc='upper right')
# plt.title('Training and validation loss')
# plt.show()

# image_batch, label_batch = next(iter(train_ds))
# prediction_batch = model.predict(image_batch)
# score_batch = tf.nn.softmax(prediction_batch)
# test_apple_url = images[0].numpy().astype("uint8")
# img = tf.keras.utils.load_img(test_apple_url, target_size=(img_height, img_width))


# solution
# img_array = tf.keras.utils.img_to_array(images[0].numpy())
# img_array = tf.expand_dims(img_array, 0)  # create a batch

# predictions_apple = model.predict(img_array)
# score_apple = tf.nn.softmax(predictions_apple[0])
# plt.subplot(1, 2, 1)
# plt.imshow(images[0].numpy().astype('uint'))
# plt.axis("on")
# plt.show()
# if(class_names[np.argmax(score_apple)][:6] == "rotten"):
#     print("This", class_names[np.argmax(score_apple)][6:], " is {:.2f}".format(
#         100-(100 * np.max(score_apple))), "% healthy")


# else:
#     print("This", class_names[np.argmax(score_apple)][5:], " is {:.2f}".format(
#         100 * np.max(score_apple)), "% healthy")
