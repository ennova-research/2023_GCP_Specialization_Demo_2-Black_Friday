from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn

import json
import numpy as np
import pandas as pd
import pickle
from google.cloud import storage
import os

import Demo2

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="google_credentials.json"

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
def preprocess(data_path: str, processed_data_path: str, background_tasks: BackgroundTasks, threshold: float = None):
    """
    Endpoint for preprocessing data.

    Args:
        data_path (str): Path to the input CSV file.
        processed_data_path (str): Path to save the preprocessed data.
        threshold (float, optional): Threshold value for preprocessing. Default is None.

    Returns:
        JSONResponse: JSON response containing the path to the preprocessed data.
    """
    def preprocess_data_task():
        try:
            # Read input data from CSV file
            data = pd.read_csv(data_path)

            # Perform data preprocessing
            preprocessed_data = preprocess_data(data, threshold)

            # Save preprocessed data to a binary file using pickle
            with open(processed_data_path, 'wb') as f:
                pickle.dump(preprocessed_data, f)

            # Return success response with the path to the preprocessed data
            return {"preprocessed_data_path": processed_data_path}

        except FileNotFoundError:
            # Handle file not found error
            raise HTTPException(status_code=404, detail="File not found")

        except Exception as e:
            # Handle other exceptions and return a generic 500 internal server error
            raise HTTPException(status_code=500, detail=str(e))

    # Add the preprocessing function as a background task
    background_tasks.add_task(preprocess_data_task)

    # Return an acknowledgement that the preprocessing process has been started
    return {"message": "Data preprocessing process has been started in the background."}

@app.post("/hyperparameter_tuning")
async def tune(data_path: str, background_tasks: BackgroundTasks, n_trials: int = 100, direction: str = 'maximize', preprocess: bool = True):
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
    async def hyperparameter_tuning_task(data_path: str, n_trials: int, direction: str, preprocess: bool):
        try:
            # Read data from CSV file
            data = pd.read_csv(data_path)

            # Preprocess data if preprocess flag is True
            if preprocess:
                data = preprocess_data(data)

            # Perform hyperparameter tuning
            study = tune(data['X'], data['y'], n_trials, direction)

            # Log or save the best hyperparameters, or perform other actions as needed
            # TODO: IMPLEMENT SAVING OF THIS
            print(study)

        except Exception as e:
            # Log or handle exceptions here
            print(str(e))

    # Enqueue hyperparameter tuning task as a background task
    background_tasks.add_task(hyperparameter_tuning_task, data_path, n_trials, direction, preprocess)
    
    # Return a response indicating that the task has been enqueued
    return JSONResponse(content={"message": "Hyperparameter tuning task has been enqueued."}, status_code=200)


@app.post("/train")
def train(final_model_path, background_tasks:BackgroundTasks, initial_model_path=None, model_params={}, data_path=None, preprocess=True):
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
    def train_model():
        try:
            # Load the initial model or create a new one
            if initial_model_path:
                with open(initial_model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                if isinstance(model_params, str):
                    model_params = json.loads(model_params)
                model = Demo2.training.create_model(model_params)

            # Load training data and preprocess if required
            if data_path:
                data = pd.read_csv(data_path)
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
            with open(final_model_path, 'wb') as f:
                pickle.dump(model, f)

            # Return a JSON response with relevant information
            # TODO: IMPLEMENT SAVING OF THIS
            return {"trained_model_path": final_model_path,
                     'threshold': best_threshold,
                     'score': scores.mean()}
        except Exception as e:
            # Handle exceptions and return a meaningful error response
            raise HTTPException(status_code=500, detail=str(e))
     # Add the training function as a background task
    background_tasks.add_task(train_model)
    
    # Return an acknowledgement that the training process has been started
    return {"message": "Model training process has been started in the background."}

    


@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint to make predictions based on input data.

    Args:
        data_path (str): Path to the CSV file containing input data.
        model (object): The machine learning model for prediction.
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
        model = body["model"]

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