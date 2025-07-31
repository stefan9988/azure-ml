
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Environment

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

ml_client = MLClient.from_config(credential=credential)

try:
    custom_env = ml_client.environments.get(name="sklearn-env-custom-training", version="1")
except Exception:
    custom_env = Environment(
        name="sklearn-env-custom-training",
        conda_file="conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version="1"
    )

    ml_client.environments.create_or_update(custom_env)

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

try:
    data_asset = ml_client.data.get(name="diabetes-file", version="1")
except Exception:
    my_data = Data(
        path="diabetes-data/diabetes.csv",
        type=AssetTypes.URI_FILE,         
        name="diabetes-file",             
        version="1"
    )

    ml_client.data.create_or_update(my_data)


job = command(
    code="./src",
    command="python custom_training_script.py --training_data ${{inputs.input_data}}",
    inputs={
        "input_data": Input(
            type="uri_file",
            path="azureml:diabetes-file:1"
        )        
    },
    environment="sklearn-env-custom-training:1",
    compute="aml-cluster",
    display_name="diabetes-train-autolog",
    experiment_name="diabetes-custom-training"
    )

returned_job = ml_client.create_or_update(job)
