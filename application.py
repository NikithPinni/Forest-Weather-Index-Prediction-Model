from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods = ["GET","POST"])
def predict_datapoint():
    if(request.method == "POST"):
        # The below lines are for reading the values entered in html file and storing in pythonic variables
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        # Note the point that the string in get function in the above code block must 
        # be same as name attribute in input tag in home.html file
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template("home.html",result = result[0])
        # The above pythonic variable results must be same as variable in ln23 in home.html file 
    
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0" , port = 8080)
