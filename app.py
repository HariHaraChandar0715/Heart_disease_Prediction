import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

### Initiating the app
app=Flask(__name__)
### Load the model
logicmodel=pickle.load(open("logisticmodel.pkl","rb"))
### Home Page
@app.route('/')
def home():
    return render_template('home.html')


### Creating the predict API
'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    predoutput=regmodel.predict(new_data)
    print(predoutput[0])
    return jsonify(predoutput[0]) 
'''
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    #print(final_input)
    output=logicmodel.predict(final_input)[0]
    if(output == 1):
          return render_template("home.html",prediction_text="OOPS you are Affected by Heart Disease")
    else:
          return render_template("home.html",prediction_text="You Are Not Affected")      

    ##return render_template("home.html",prediction_text="The Predicted House Price is :  {}".format(output))

if __name__=="__main__":
    app.run(port=5000,debug=True)