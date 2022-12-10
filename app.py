from flask import Flask,request,render_template,Response
import prediction
import pandas as pd

app=Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    predicted_output=" "
    if request.files["data"].filename!='':
        predict=prediction.predict_value()
        data=pd.read_csv(request.files["data"])
        result=predict.predict_from_file(data)
        predicted_output="Predictions are stored in CSV file"
    # else:
        # print("recieved input from number fields")
        
    return render_template('index.html',prediction_text=predicted_output)


if __name__=="__main__":
    app.run(debug=True)