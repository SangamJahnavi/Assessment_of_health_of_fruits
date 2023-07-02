from flask import Flask,render_template,url_for, request, redirect
import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


app = Flask(__name__)

modelp = pickle.load(open('model.pkl', 'rb'))

def pred_label(test_apple_url):
    class_names = ['freshapples', 'freshbanana', 'freshoranges',
               'rottenapples', 'rottenbanana', 'rottenoranges']
    img = tf.keras.utils.load_img(
    test_apple_url, target_size=(100, 100))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # create a batch

    predictions_apple = modelp.predict(img_array)
    score_apple = tf.nn.softmax(predictions_apple[0])

    if(class_names[np.argmax(score_apple)][:6] == "rotten"):
        print("This", class_names[np.argmax(score_apple)][6:], " is {:.2f}".format(
        100-(100 * np.max(score_apple))), "% healthy")
        formatted_number = format(100-(100 * np.max(score_apple)), ".2f")
        return formatted_number,class_names[np.argmax(score_apple)]



    else:
        print("This", class_names[np.argmax(score_apple)][5:], " is {:.2f}".format(
        100 * np.max(score_apple)), "% healthy")
        formatted_number = format(100 * np.max(score_apple), ".2f")
        return formatted_number,class_names[np.argmax(score_apple)]


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/submit',methods=['GET','POST'])
def output():
    if request.method=='POST':
        img=request.files['fruitimage']
        img_path="static/"+img.filename
        img.save(img_path)
        p=pred_label(img_path)
        return render_template("index.html",prediction=p,img_path=img_path)
    


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0",port=5000)
