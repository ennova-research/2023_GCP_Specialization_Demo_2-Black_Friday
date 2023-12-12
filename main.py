from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn

import numpy as np
import pandas as pd
import pickle
from google.cloud import storage

import Demo2


app = FastAPI()

project_id = 'ml-spec'
bucket_name = 'engo-ml_spec2023-demo2'
blob_name = 'outputs.pkl'
client = storage.Client(project=project_id)
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)
serialized_data = blob.download_as_text()
my_dict = pickle.loads(serialized_data)
model = my_dict['model']
threshold = my_dict['threshold']
# with open('./tmp/model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('./tmp/best_threshold.pkl', 'rb') as f:
#     threshold = pickle.load(f)


def preprocess_data(data, threshold = None):
    users = Demo2.eda.group_by_user(data)
    if 'Sum spent' not in users.columns:
        users['Sum spent'] = np.zeros_like(users.index)
    
    X_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    X, y, threshold = Demo2.preprocess_per_classification(users, X_cols=X_cols, old_y_col='Sum spent', new_y_col='is_BigSpender', threshold=threshold)
    return {'X': X, 'y': y, 'threshold': threshold}


def predict(model, X, threshold = None):
    if threshold:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= float(threshold)).astype(int)
    else:
        y_pred = model.predict(X)
    return y_pred

@app.get("/")
def read_root():
    return "GCP Specialization - Demo 2"


@app.post("/preprocess")
def preprocess(data_path, processed_data_path, threshold=None):
    try:
        data = pd.read_csv(data_path)
        preprocessed_data = preprocess_data(data, threshold)
        with open(processed_data_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        return JSONResponse(content={"preprocessed_data_path": processed_data_path}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hyperparameter_tuning")
def tune(data_path, n_trials=100, direction='maximize', preprocess=True):
    try:
        data = pd.read_csv(data_path)
        if preprocess:
            data = preprocess_data(data)
        study = Demo2.training.tune(data['X'], data['y'], n_trials, direction)
        return JSONResponse(content={"params": study.best_params}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/train")
def train(final_model_path, initial_model_path=None, model_params={}, data_path=None, preprocess=True):
    try:
        if initial_model_path:
            with open(initial_model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = Demo2.training.create_model(model_params)
        if data_path:
            data = pd.read_csv(data_path)
            if preprocess:
                data = preprocess_data(data)
        else:
            data = pd.read_csv('gs://engo-ml_spec2023-demo2/data_raw.csv')
            data = preprocess_data(data)
            
        model.fit(data['X'], data['y'])
        best_threshold = Demo2.training.select_threshold(model, data['X'], data['y'])
        scores = Demo2.training.evaluate(model, data['X'], data['y'], classification_threshold=best_threshold, cross_val=True)
        with open(final_model_path, 'wb') as f:
            pickle.dump(model, f)
        return JSONResponse(content={"trained_model_path": final_model_path,
                                     'threshold': best_threshold,
                                     'score': scores.mean()}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/predict")
def make_prediction(data_path, model = model, threshold = threshold, preprocess = True):
    try:
        data = pd.read_csv(data_path)
        if preprocess:
            data = preprocess_data(data)
        predictions = predict(model, data['X'], threshold).tolist()
        return JSONResponse(content={"predictions": predictions}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host="0.0.0.0")#, reload=True)