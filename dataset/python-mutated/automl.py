"""This module contains Google AutoML links."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.utils.context import Context
AUTOML_BASE_LINK = 'https://console.cloud.google.com/automl-tables'
AUTOML_DATASET_LINK = AUTOML_BASE_LINK + '/locations/{location}/datasets/{dataset_id}/schemav2?project={project_id}'
AUTOML_DATASET_LIST_LINK = AUTOML_BASE_LINK + '/datasets?project={project_id}'
AUTOML_MODEL_LINK = AUTOML_BASE_LINK + '/locations/{location}/datasets/{dataset_id};modelId={model_id}/evaluate?project={project_id}'
AUTOML_MODEL_TRAIN_LINK = AUTOML_BASE_LINK + '/locations/{location}/datasets/{dataset_id}/train?project={project_id}'
AUTOML_MODEL_PREDICT_LINK = AUTOML_BASE_LINK + '/locations/{location}/datasets/{dataset_id};modelId={model_id}/predict?project={project_id}'

class AutoMLDatasetLink(BaseGoogleLink):
    """Helper class for constructing AutoML Dataset link."""
    name = 'AutoML Dataset'
    key = 'automl_dataset'
    format_str = AUTOML_DATASET_LINK

    @staticmethod
    def persist(context: Context, task_instance, dataset_id: str, project_id: str):
        if False:
            i = 10
            return i + 15
        task_instance.xcom_push(context, key=AutoMLDatasetLink.key, value={'location': task_instance.location, 'dataset_id': dataset_id, 'project_id': project_id})

class AutoMLDatasetListLink(BaseGoogleLink):
    """Helper class for constructing AutoML Dataset List link."""
    name = 'AutoML Dataset List'
    key = 'automl_dataset_list'
    format_str = AUTOML_DATASET_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            return 10
        task_instance.xcom_push(context, key=AutoMLDatasetListLink.key, value={'project_id': project_id})

class AutoMLModelLink(BaseGoogleLink):
    """Helper class for constructing AutoML Model link."""
    name = 'AutoML Model'
    key = 'automl_model'
    format_str = AUTOML_MODEL_LINK

    @staticmethod
    def persist(context: Context, task_instance, dataset_id: str, model_id: str, project_id: str):
        if False:
            while True:
                i = 10
        task_instance.xcom_push(context, key=AutoMLModelLink.key, value={'location': task_instance.location, 'dataset_id': dataset_id, 'model_id': model_id, 'project_id': project_id})

class AutoMLModelTrainLink(BaseGoogleLink):
    """Helper class for constructing AutoML Model Train link."""
    name = 'AutoML Model Train'
    key = 'automl_model_train'
    format_str = AUTOML_MODEL_TRAIN_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            return 10
        task_instance.xcom_push(context, key=AutoMLModelTrainLink.key, value={'location': task_instance.location, 'dataset_id': task_instance.model['dataset_id'], 'project_id': project_id})

class AutoMLModelPredictLink(BaseGoogleLink):
    """Helper class for constructing AutoML Model Predict link."""
    name = 'AutoML Model Predict'
    key = 'automl_model_predict'
    format_str = AUTOML_MODEL_PREDICT_LINK

    @staticmethod
    def persist(context: Context, task_instance, model_id: str, project_id: str):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context, key=AutoMLModelPredictLink.key, value={'location': task_instance.location, 'dataset_id': '-', 'model_id': model_id, 'project_id': project_id})