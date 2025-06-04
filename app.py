import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model = load('random_forest_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
   
    x_test = [[int(x) for x in request.form.values()]]
    print(x_test)
    if(x_test[0][1] == 0):
        x_test[0][1]=0
        x_test[0].insert(2,0)
    elif(x_test[0][1] == 1):
        x_test[0][1]=0
        x_test[0].insert(2,1)
    else:
        x_test[0][1]=1
        x_test[0].insert(2,0)
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=(prediction[0])
    chance = ''
    if output == 0:
        chance = "Low"
    elif output == 1:
        chance = "Medium"
    else:
        chance = "High"
    return render_template('./index.html', prediction_text='Avalanche Category-{}. It has a {} chance'.format(output,chance))

@app.route('/predict_api',methods=['POST'])
def predict_api():
  
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
