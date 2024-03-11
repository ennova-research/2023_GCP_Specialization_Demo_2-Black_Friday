# 2023 - GCP ML Specialization
## Demo 2: Black_Friday

This demo demonstrate how to predict the spending power of clients during Blak Friday. In the demo we explored both a regression type of prediction (how much each client spent) and a classification one (is the client a "big spender"?). We decided to follow the latter, but the possibility to switch to regression was left in the code, to allow an easy change in future if needed.

The project contains a directory `notebooks` containing the following scripts:
1. `0_eda.ipynb`, containing the Exploratory Data Analysis performed.
2. `1_preprocess.ipynb`, which contains the procedure followed for preprocessing the data.
3. `2_training.ipynb`, containing the hyperparameter tuning for the model and the training phase.
4. `3_evaluation.ipynb`, which contains the evaluation procedure for the model on the test data.
5. `pipeline.ipynb`, that is the union of all the previus notebooks, to have an unified view.

The `Demo2` directory contains the library with all the functions used by this demo.

Finally, the `main.py` contains the FastAPI app used to deliver the services.

As a demo, we stayed minimal, but we expect to be able to extract more info from the dataset. This could be done, e.g., by:
1. Classifing the more valuable products/product_categories.
2. Predictiong how much each user will spend for each product.


## Installation

These instructions assume that you have both Docker and Google Clour SDK on your machine, and that you have access to the Google Cloud Platform `ml-spec` project. If you don't, make sure to have met these conditions before proceeding.

The original dataset can be found at the following link:

https://www.kaggle.com/abhisingh10p14/black-friday

The "data_raw.csv" file we created in Cloud Storage ('gs://engo-ml_spec2023-demo2/data_raw.csv') and that we referenced in the code, is simply an upload of the "train.csv" we can download from the kaggle link.
We used just the "train" and not the "test" file, since in the "test" there is no column "Purchase" to use as evaluation. 

To run the code in the notebooks, we have created a conda environment to work with. In case the environment "demo2" is missing, run the following command in the terminal: `conda env create -f environment.yml`. Then create the kernel with the following commands: `conda activate demo2`, `ipython kernel install --user --name=Demo 2`. Once you have the kernel, you can work with the notebooks on Vertex AI Workbench

The app is currently deployed at the following link:

https://demo-2-b6lmpdo3cq-uc.a.run.app/docs

To deploy a new version of the app, first create a container by running:

```docker build -t demo-2 .```

Then upload the container on Google Cloud Build by running:

```gcloud builds submit --tag gcr.io/ml-spec/demo-2```

Finally deploy the container on Google Cloud Run by running:

```gcloud run deploy --image gcr.io/ml-spec/demo-2 --platform managed --port 5000 --memory 4G --cpu 2 --timeout 60m```