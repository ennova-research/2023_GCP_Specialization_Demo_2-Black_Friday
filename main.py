from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

import json
import numpy as np
import pandas as pd
import pickle
from google.cloud import storage

import os

import Demo2

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="google_credentials.json"

app = FastAPI()

# Google Cloud Storage configuration
project_id = 'ml-spec'
bucket_name = 'engo-ml_spec2023-demo2'
blob_name = 'outputs.pkl'
client = storage.Client(project=project_id)
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)

# Download serialized data from GCS
serialized_data = blob.download_as_bytes()
my_dict = pickle.loads(serialized_data)
model = my_dict['model']
threshold = my_dict['threshold']

class PreprocessRequest(BaseModel):
    data_path: str = "gs://engo-ml_spec2023-demo2/data_raw.csv"
    threshold: float = None

class HyperparameterTuningRequest(BaseModel):
    data_path: str = "gs://engo-ml_spec2023-demo2/data_raw.csv"
    n_trials: int = 100
    direction: str = 'maximize'
    preprocess: bool = True

class TrainRequest(BaseModel):
    initial_model_path: str = None
    model_params: str = "gs://engo-ml_spec2023-demo2/study_hyper.csv"
    data_path: str = None
    preprocess: bool = True

def preprocess_data(data, threshold=None):
    """
    Preprocesses input data for classification.

    Args:
        data (DataFrame): Input data containing user information.
        threshold (float, optional): Threshold value for classification. Default is None.

    Returns:
        dict: A dictionary containing preprocessed data, labels, and threshold.
            - 'X': Features for classification.
            - 'y': Labels for classification.
            - 'threshold': Threshold value used for classification.
    """
    # Group data by user using the 'eda' module from Demo2
    users = Demo2.eda.group_by_user(data)

    # Add 'Sum spent' column if not present in users DataFrame
    if 'Sum spent' not in users.columns:
        users['Sum spent'] = np.zeros_like(users.index)

    # Define columns for feature extraction
    X_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

    # Preprocess data using the 'preprocess_per_classification' function from Demo2
    X, y, threshold = Demo2.preprocess_per_classification(users, X_cols=X_cols,
                                                          old_y_col='Sum spent', new_y_col='is_BigSpender',
                                                          threshold=threshold)

    # Return the preprocessed data as a dictionary
    return {'X': X, 'y': y, 'threshold': threshold}


def make_prediction(model, X, threshold=None):
    """
    Make predictions using the given model on input data X.

    Parameters:
    - model: The predictive model to use for making predictions.
    - X: The input data for prediction.
    - threshold: (Optional) Threshold for binary classification. If provided,
                 predictions will be based on this threshold.

    Returns:
    - y_pred: Predicted labels based on the given model and threshold.
    """

    # Check if a threshold is provided
    if threshold:
        # Probability prediction for positive class
        y_prob = model.predict_proba(X)[:, 1]
        
        # Convert probabilities to binary predictions using the threshold
        y_pred = (y_prob >= float(threshold)).astype(int)
    else:
        # Use default model prediction for binary classification
        y_pred = model.predict(X)

    return y_pred


@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return "GCP Specialization - Demo 2"


@app.post("/preprocess")
async def preprocess(request:PreprocessRequest, background_tasks: BackgroundTasks):
    """
    Endpoint for preprocessing data.

    Args:
        data_path (str): Path to the input CSV file.
        processed_data_path (str): Path to save the preprocessed data.
        threshold (float, optional): Threshold value for preprocessing. Default is None.

    Returns:
        JSONResponse: JSON response containing the path to the preprocessed data.
    """
    def preprocess_data_task(request:PreprocessRequest):
        global bucket
        try:
            # Read input data from CSV file
            data = pd.read_csv(request.data_path)

            # Perform data preprocessing
            preprocessed_data = preprocess_data(data, threshold)

            # Save preprocessed data to a binary file using pickle
            with open("preprocessed_data.csv", 'wb') as f:
                pickle.dump(preprocessed_data, f)

            blob = bucket.blob("preprocessed_data.csv")
            blob.upload_from_filename("preprocessed_data.csv")
            print("Preprocessed data saved to bucket")

        except FileNotFoundError as e:
            # Handle file not found error
            print(str(e))

        except Exception as e:
            # Handle other exceptions and return a generic 500 internal server error
            print(str(e))

    # Add the preprocessing function as a background task
    background_tasks.add_task(preprocess_data_task, request)

    # Return an acknowledgement that the preprocessing process has been started
    return {"message": "Data preprocessing process has been started in the background."}


@app.post("/hyperparameter_tuning")
async def tune(request:HyperparameterTuningRequest, background_tasks: BackgroundTasks):
    """
    Endpoint for hyperparameter tuning.

    Parameters:
    - data_path (str): Path to the CSV file containing the data.
    - n_trials (int): Number of hyperparameter optimization trials (default is 100).
    - direction (str): Direction of optimization, either 'maximize' or 'minimize' (default is 'maximize').
    - preprocess (bool): Flag indicating whether to preprocess the data (default is True).

    Returns:
    - JSONResponse: JSON response containing the best hyperparameters.

    Raises:
    - HTTPException: If an error occurs during the execution.
    """
    def hyperparameter_tuning_task(request:HyperparameterTuningRequest):
        global bucket
        try:
            # Read data from CSV file
            data = pd.read_csv(request.data_path)

            # Preprocess data if preprocess flag is True
            if request.preprocess:
                data = preprocess_data(data)

            # Perform hyperparameter tuning
            study = Demo2.training.tune(data['X'], data['y'], request.n_trials, request.direction)

            # Log or save the best hyperparameters, or perform other actions as needed
             # Return a JSON response with relevant information
            
            with open ("study_hyper.json", 'w') as fp:
                json.dump(study, fp)

            blob = bucket.blob("study_hyper.json")
            blob.upload_from_filename("study_hyper.json")
            print("study_hyper.json uploaded to bucket")

        except Exception as e:
            print(str(e))

    # Enqueue hyperparameter tuning task as a background task
    background_tasks.add_task(hyperparameter_tuning_task, request)
    
    # Return a response indicating that the task has been enqueued
    return JSONResponse(content={"message": "Hyperparameter tuning task has been enqueued."}, status_code=200)


@app.post("/train")
async def train(request:TrainRequest, background_tasks:BackgroundTasks):
    """
    Endpoint for training a machine learning model.

    Parameters:
    - final_model_path (str): Path to save the trained model.
    - initial_model_path (str, optional): Path to the initial model for transfer learning.
    - model_params (dict, optional): Parameters for creating the model.
    - data_path (str, optional): Path to the training data CSV file.
    - preprocess (bool, optional): Flag indicating whether to preprocess the data.

    Returns:
    - JSONResponse: JSON response containing information about the trained model, threshold, and score.
    """
    def train_model(request:TrainRequest):
        global bucket
        try:
            # Load the initial model or create a new one
            if request.initial_model_path:
                with open(request.initial_model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                if isinstance(request.model_params, str):
                    #TODO check if correct parameters
                    blob = storage.Blob("study_hyper.json", bucket)
                    with open("study_hyper_from_gs.json", "wb") as file_obj:
                        blob.download_to_file(file_obj)
                    
                    with open("study_hyper_from_gs.json", 'r') as model_parameters_json:
                        model_parameters = json.load(model_parameters_json)
                        model = Demo2.training.create_model(model_parameters)

            # Load training data and preprocess if required
            if request.data_path:
                data = pd.read_csv(request.data_path)
                if preprocess:
                    data = preprocess_data(data)
            else:
                # Use default data if no path provided
                data = pd.read_csv('gs://engo-ml_spec2023-demo2/data_raw.csv')
                data = preprocess_data(data)

            # Fit the model and select the best threshold
            model.fit(data['X'], data['y'])
            best_threshold = Demo2.training.select_threshold(model, data['X'], data['y'])

            # Evaluate the model and get scores
            scores = Demo2.training.evaluate(model, data['X'], data['y'], classification_threshold=best_threshold, cross_val=True)

            # Save the trained model
            with open("model.pkl", 'wb') as f:
                pickle.dump(model, f)

            blob = bucket.blob("model.pkl")
            blob.upload_from_filename("model.pkl")
            print("model.pkl uploaded to bucket")

            # Return a JSON response with relevant information
            
            with open ("scores.json", 'w') as fp:
                json.dump({ 
                     'threshold': best_threshold,
                     'score': scores.mean()}, fp)

            blob = bucket.blob("scores.json")
            blob.upload_from_filename("scores.json")
            print("scores.json uploaded to bucket")

        except Exception as e:
            # Handle exceptions and return a meaningful error response
            print(str(e))

     # Add the training function as a background task
    background_tasks.add_task(train_model, request)
    
    # Return an acknowledgement that the training process has been started
    return {"message": "Model training process has been started in the background."}


@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint to make predictions based on input data.

    Args:
        data_path (str): Path to the CSV file containing input data.
        model_path (str): The path where the machine learning model for prediction is.
        threshold (float): Threshold for the prediction.
        preprocess (bool): Flag indicating whether to preprocess the data.

    Returns:
        JSONResponse: JSON response containing the predictions.
    """

    global model, threshold, preprocess_field

    req_data = await request.json()
    body = req_data["instances"][0]

    if "model" not in body:
        model = model
    else:
        #read model from gs://
        blob = storage.Blob(body.model_path.split("/")[-1], bucket)
        with open("tmp_model.pkl", "wb") as file_obj:
            blob.download_to_file(file_obj)
        model = pickle.load(open("tmp_model.pkl", "rb"))

    if "threshold" not in body:
        threshold=threshold
    else:
        threshold=body["threshold"]

    if "preprocess" not in body:
        preprocess_field=True
    else:
        preprocess_field=body["preprocess"]

    if "data_path" not in body:
        raise HTTPException(500, "malformed request. data path field required")

    try:
        # Read input data from CSV file
        data = pd.read_csv(body["data_path"])

        # Perform data preprocessing if requested
        if preprocess_field:
            data = preprocess_data(data)

        # Make predictions using the provided model and threshold
        predictions = make_prediction(model, data['X'], threshold).tolist()

        # Return predictions as JSON response
        return {"predictions": predictions}

    except Exception as e:
        # Handle exceptions and return an HTTP 500 error with details
        print()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "alive"}


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host="0.0.0.0")#, reload=True)
