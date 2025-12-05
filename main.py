import pandas as pd
from fastapi import FastAPI #import FastAPI
import joblib #import the crocodile classification ML model
from pydantic import BaseModel #BaseModel makes us input a specific input type orelse it validates an error
from typing import Optional 
from fastapi import HTTPException

app=FastAPI()

model=joblib.load('model.pkl')

class InputData(BaseModel):
    Observed_Length : float
    Observed_Weight: float
    Age_Class: str
    Sex: Optional[str] = None #Make Sex variable optional, the model takes "Unknown" sex for variable
    Country_or_Region : str
    Habitat_type: str
    Conservation_status: str

@app.post('/predict')
def predict(data:InputData):
    try:
        df=pd.DataFrame([{
            "Observed Length (m)": data.Observed_Length,
            "Observed Weight (kg)":data.Observed_Weight,
            "Age Class": data.Age_Class,
            "Sex":data.Sex,
            "Country/Region":data.Country_or_Region,
            "Habitat Type": data.Habitat_type,
            "Conservation Status":data.Conservation_status
        }])
        #the problems that I ran into:
        # "columns are missing: {'Conservation Status', 'Country/Region', 'Observed Length (m)', 'Age Class', 'Observed Weight (kg)', 'Habitat Type'}"
        # "Specifying the columns using strings is only supported for dataframes."


        prediction=model.predict(df)[0]
        return{'prediction':prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

