from flask import Flask,request,render_template,Response

import pandas as pd

app=Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    if request.files["data"].filename!='':
        print("recieved input from file")
    else:
        print("recieved input from number fields")
        
    return render_template('index.html',prediction_text="hi")


if __name__=="__main__":
    app.run(debug=True)