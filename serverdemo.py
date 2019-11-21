import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    uinput = [x for x in request.form.values()]
    print(uinput)
    # create input features for model
    minput=[]
    minput.append(float(uinput[3]))
    minput.append(float(uinput[4]))
    minput.append(int(uinput[5]))
    minput.extend([1,0,0])
    if uinput[7] == 'Bronx':
        minput.extend([1,0,0,0,0])
    elif uinput[7] == 'Brooklyn':
        minput.extend([0,1,0,0,0])
    elif uinput[7] == 'Manhattan':
        minput.extend([0,0,1,0,0])
    elif uinput[7] == 'Queens':
        minput.extend([0,0,0,1,0])
    else:
        minput.extend([0,0,0,0,1])

    if uinput[8] == 'Entire home/apt':
        minput.extend([1,0,0])
    elif uinput[8] == 'Private room':
        minput.extend([0,1,0])
    else:
        minput.extend([0,0,1])

    modelname="model.{}.pkl".format(uinput[9])
    model = pickle.load(open(modelname,'rb'))
    pinput = []
    pinput.append(minput)
    print(pinput)
    
    price = model.predict(pinput)
    print(price)
    output = price[0]

    return render_template('index.html', prediction_text='The recommended price is: {0:.2f}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    uinput = request.get_json(force=True)
    minput=[]
    minput.append(uinput['latitude'])
    minput.append(uinput['longitude'])
    minput.append(uinput['minimum_nights'])
    minput.extend([1,0,0])
    if uinput['neighbourhood_group'] == 'Bronx':
        minput.extend([1,0,0,0,0])
    elif uinput['neighbourhood_group'] == 'Brooklyn':
        minput.extend([0,1,0,0,0])
    elif uinput['neighbourhood_group'] == 'Manhattan':
        minput.extend([0,0,1,0,0])
    elif uinput['neighbourhood_group'] == 'Queens':
        minput.extend([0,0,0,1,0])
    else:
        minput.extend([0,0,0,0,1])

    if uinput['room_type'] == 'Entire home/apt':
        minput.extend([1,0,0])
    elif uinput['room_type'] == 'Private room':
        minput.extend([0,1,0])
    else:
        minput.extend([0,0,1])

    modelname="model.{}.pkl".format(uinput['vacancy threshold'])
    model = pickle.load(open(modelname,'rb'))
    pinput = []
    pinput.append(minput)
    price = model.predict(pinput)
    print(price)
    output = price[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(port=33333,debug=True)