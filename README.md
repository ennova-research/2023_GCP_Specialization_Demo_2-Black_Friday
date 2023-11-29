# 2023_GCP_Specialization_Demo_2-Black_Friday
Google Cloud Partner Advantage 

2023 Specialization 

3.0 Capabilities Assesment 

Demo #2 - Black Friday 

data: https://www.kaggle.com/abhisingh10p14/black-friday
The "data_raw.csv" file we created in Cloud Storage ('gs://engo-ml_spec2023-demo2/data_raw.csv') is simply an upload of the "train.csv" we can download from the kaggle link.
We used just the "train" and not the "test" file, since in the "test" there is no column "Purchase" to use as evaluation. 

We have created a conda environment to work with. In case the environment "demo2" is missing, run the following command in the terminal: `conda env create -f environment.yml`. Then create the kernel with the following commands: 
`conda activate demo2` 
`ipython kernel install --user --name=Demo 2` 
Once you have the kernel, you can weork with the notebooks on Vertex AI Workbench