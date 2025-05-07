from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app_name=Flask(__name__)
app=app_name

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_data",methods=["GET","POST"])
def predict_data():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race"),
            parental_level_of_education=request.form.get("edu"),
            test_preparation_course=request.form.get("test"),
            lunch=request.form.get("Lunch"),
            writing_score=float(request.form.get("Writing Score")),
            reading_score=float(request.form.get("Reading Score"))
        )
        pred_df=data.convert_to_df()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=result[0])
    
if __name__=="__main__":
    print("App Started")   
    app.run(debug=True) 
    