from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import automl

DATA_ASSET_NAME = "diabetes"
AML_CLUSTER_NAME = "aml-cluster"

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # credential = InteractiveBrowserCredential() 
    pass   

ml_client = MLClient.from_config(credential=credential)
     
ws = ml_client.workspaces.get(name=ml_client.workspace_name)

try:
    data_asset = ml_client.data.get(name=DATA_ASSET_NAME, version="3")
except Exception:
    local_path = 'diabetes-data/'
    my_data = Data(
        path=local_path,
        type=AssetTypes.MLTABLE,
        name=DATA_ASSET_NAME,
        version="3"
    )

    ml_client.data.create_or_update(my_data)

try:
    aml_cluster = ml_client.compute.get(name=AML_CLUSTER_NAME)
except Exception:
    aml_cluster = AmlCompute(
        name=AML_CLUSTER_NAME,
        type="amlcompute",
        size="Standard_DS3_v2",
        location=ws.location,
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=60,
        tier="Dedicated"
    )

    ml_client.begin_create_or_update(aml_cluster).result()

classification_job = automl.classification(
    compute=aml_cluster.name,  
    experiment_name="diabetes-automl-classification",
    training_data=Input(type=AssetTypes.MLTABLE, path=f"{data_asset.id}"),
    target_column_name="Diabetic",  
    primary_metric="accuracy",  
    n_cross_validations=5,
    enable_model_explainability=True
)

classification_job.set_limits(
    max_trials=4,
    timeout_minutes=60,
    trial_timeout_minutes=30,
    enable_early_termination=True,
)

returned_job = ml_client.jobs.create_or_update(classification_job)