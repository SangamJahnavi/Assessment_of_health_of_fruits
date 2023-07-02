import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
modelp = pickle.load(open('model.pkl', 'rb'))

class_names = ['freshapples', 'freshbanana', 'freshoranges',
               'rottenapples', 'rottenbanana', 'rottenoranges']
test_apple_url = r'test_dataset\rotated_by_15_Screen Shot 2018-06-07 at 2.34.18 PM.png'
img = tf.keras.utils.load_img(
    test_apple_url, target_size=(100, 100))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # create a batch

predictions_apple = modelp.predict(img_array)
score_apple = tf.nn.softmax(predictions_apple[0])
# plt.subplot(1, 2, 1)
# plt.imshow(
#     'rotated_by_15_Screen Shot 2018-06-12 at 9.38.04 PM.png')
# plt.axis("on")
# plt.show()
if(class_names[np.argmax(score_apple)][:6] == "rotten"):
    print("This", class_names[np.argmax(score_apple)][6:], " is {:.2f}".format(
        100-(100 * np.max(score_apple))), "% healthy")


else:
    print("This", class_names[np.argmax(score_apple)][5:], " is {:.2f}".format(
        100 * np.max(score_apple)), "% healthy")
