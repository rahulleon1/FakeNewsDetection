import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
#import nltk
import re
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from scikit import predict_classes
from sklearn.model_selection import train_test_split
from flask import Flask,redirect,render_template,request,flash,url_for


voc_size=5000
sent_length=20
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())
with open('fake_x_data_political.data', 'rb') as filehandle:
    # read the data as binary data stream
    x_from_pickle = pickle.load(filehandle)


with open('fake_y_data_political.data', 'rb') as filehandle2:
    # read the data as binary data stream
    y_from_pickle = pickle.load(filehandle2)


X_final=np.array(x_from_pickle)
y_final=np.array(y_from_pickle)


X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('fakenews.html')


@app.route('/index.html')
def index():
    return render_template('fakenews.html')


@app.route('/fakenews.html')
def fake():
    return render_template('fakenews.html')


@app.route('/fakenews', methods=['POST', 'GET'])
def fakenews():
    if request.method == 'POST':

        input_from_user = request.form['fakenews']
        ps = PorterStemmer()
        input_data = []

        print("Applying regex")
        input_sample = re.sub('[^a-zA-Z]', ' ', input_from_user)
        input_sample = input_sample.lower()
        input_sample = input_sample.split()
        print("detecting stop words")
        input_sample = [ps.stem(word) for word in input_sample if not word in stopwords.words('english')]
        input_sample = ' '.join(input_sample)
        input_data.append(input_sample)

        onehot_repr_input_data = [one_hot(words, voc_size) for words in input_data]
        sent_length = 20
        embedded_docs_input_data = pad_sequences(onehot_repr_input_data, padding='pre', maxlen=sent_length)
        X_final_input_data = np.array(embedded_docs_input_data)
        y_pred_input_data = model1.predict(X_final_input_data)
        proba = y_pred_input_data[0][0]
        if proba < 0.5:
            proba = 1 - proba
        proba = round(proba, 2)
        print(proba)
        y_pred_inputdata = predict_classes(input_from_user)

        print(y_pred_input_data)
        print(y_pred_input_data[0][0])

        if y_pred_inputdata:

            output = "fake news" if y_pred_inputdata <= 0.7 else "True news"
        else:

            output_ = "fake news" if y_pred_input_data[0][0] <= 0.7 else "True news"

        return render_template('/fakenews.html', s1=output)
    else:
        return render_template('fakenews.html')


if __name__ == '__main__':
    app.run(debug=True)


