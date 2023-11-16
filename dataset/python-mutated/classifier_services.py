"""Services for classifier data models."""
from __future__ import annotations
import base64
import datetime
import hashlib
import hmac
import logging
from core import feconf
from core.domain import classifier_domain
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import fs_services
from core.domain import state_domain
from core.platform import models
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union
MYPY = False
if MYPY:
    from mypy_imports import classifier_models
    from mypy_imports import secrets_services
    from proto_files import text_classifier_pb2
secrets_services = models.Registry.import_secrets_services()
(classifier_models,) = models.Registry.import_models([models.Names.CLASSIFIER])

class JobInfoDict(TypedDict):
    """Type for the job info dictionary."""
    algorithm_id: str
    interaction_id: str
    exp_id: str
    exp_version: int
    next_scheduled_check_time: datetime.datetime
    state_name: str
    training_data: List[state_domain.TrainingDataDict]
    status: str
    algorithm_version: int

def generate_signature(secret: bytes, message: bytes, vm_id: str) -> str:
    if False:
        i = 10
        return i + 15
    'Generates digital signature for given data.\n\n    Args:\n        secret: bytes. The secret used to communicate with Oppia-ml.\n        message: bytes. The message payload data.\n        vm_id: str. The ID of the VM that generated the message.\n\n    Returns:\n        str. The signature of the payload data.\n    '
    converted_vm_id = vm_id.encode('utf-8')
    if isinstance(message, str):
        message = message.encode('utf-8')
    message = b'%s|%s' % (base64.b64encode(message), converted_vm_id)
    return hmac.new(secret, msg=message, digestmod=hashlib.sha256).hexdigest()

def verify_signature(oppia_ml_auth_info: classifier_domain.OppiaMLAuthInfo) -> bool:
    if False:
        i = 10
        return i + 15
    'Function that checks if the signature received from the VM is valid.\n\n    Args:\n        oppia_ml_auth_info: OppiaMLAuthInfo. Domain object containing\n            authentication information.\n\n    Returns:\n        bool. Whether the incoming request is valid.\n    '
    secret = None
    vm_id = secrets_services.get_secret('VM_ID')
    shared_secret_key = secrets_services.get_secret('SHARED_SECRET_KEY')
    assert shared_secret_key is not None
    assert vm_id is not None
    if vm_id == oppia_ml_auth_info.vm_id:
        secret = shared_secret_key.encode('utf-8')
    if secret is None:
        return False
    generated_signature = generate_signature(secret, oppia_ml_auth_info.message, oppia_ml_auth_info.vm_id)
    if generated_signature != oppia_ml_auth_info.signature:
        return False
    return True

def get_new_job_models_for_trainable_states(exploration: exp_domain.Exploration, state_names: List[str]) -> List[Union[classifier_models.StateTrainingJobsMappingModel, classifier_models.ClassifierTrainingJobModel]]:
    if False:
        while True:
            i = 10
    'Creates ClassifierTrainingJobModel instances for all the state names\n    passed into the function. If this function is called with version number 1,\n    we are creating jobs for all trainable states in the exploration. Otherwise,\n    a new job is being created for the states where retraining is required.\n    Note that this does not actually create models in the datastore. It just\n    creates instances of the models and returns them. The caller of this method\n    is responsible for the put operation.\n\n    Args:\n        exploration: Exploration. The Exploration domain object.\n        state_names: list(str). List of state names.\n\n    Raises:\n        Exception. No classifier algorithm found for the given interaction id.\n\n    Returns:\n        list(ClassifierTrainingJobModel|StateTrainingJobsMappingModel). The list\n        of job models corresponding to trainable states in the exploration.\n    '
    models_to_put: List[Union[classifier_models.ClassifierTrainingJobModel, classifier_models.StateTrainingJobsMappingModel]] = []
    job_dicts_list: List[JobInfoDict] = []
    exp_id = exploration.id
    exp_version = exploration.version
    for state_name in state_names:
        state = exploration.states[state_name]
        training_data = state.get_training_data()
        interaction_id = state.interaction.id
        if interaction_id not in feconf.INTERACTION_CLASSIFIER_MAPPING:
            raise Exception('No classifier algorithm found for %s interaction' % interaction_id)
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id']
        next_scheduled_check_time = datetime.datetime.utcnow()
        algorithm_version = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_version']
        dummy_classifier_training_job = classifier_domain.ClassifierTrainingJob('job_id_dummy', algorithm_id, interaction_id, exp_id, exp_version, next_scheduled_check_time, state_name, feconf.TRAINING_JOB_STATUS_NEW, training_data, algorithm_version)
        dummy_classifier_training_job.validate()
        job_dicts_list.append({'algorithm_id': algorithm_id, 'interaction_id': interaction_id, 'exp_id': exp_id, 'exp_version': exp_version, 'next_scheduled_check_time': next_scheduled_check_time, 'state_name': state_name, 'training_data': training_data, 'status': feconf.TRAINING_JOB_STATUS_NEW, 'algorithm_version': algorithm_version})
    job_models = []
    job_ids = []
    for job_dict in job_dicts_list:
        instance_id = classifier_models.ClassifierTrainingJobModel.generate_id(job_dict['exp_id'])
        training_job_instance = classifier_models.ClassifierTrainingJobModel(id=instance_id, algorithm_id=job_dict['algorithm_id'], interaction_id=job_dict['interaction_id'], exp_id=job_dict['exp_id'], exp_version=job_dict['exp_version'], next_scheduled_check_time=job_dict['next_scheduled_check_time'], state_name=job_dict['state_name'], status=job_dict['status'], training_data=job_dict['training_data'], algorithm_version=job_dict['algorithm_version'])
        job_models.append(training_job_instance)
        job_ids.append(instance_id)
    models_to_put.extend(job_models)
    state_training_jobs_mappings = []
    for (job_id_index, job_id) in enumerate(job_ids):
        state_training_jobs_mapping = classifier_domain.StateTrainingJobsMapping(job_dicts_list[job_id_index]['exp_id'], job_dicts_list[job_id_index]['exp_version'], job_dicts_list[job_id_index]['state_name'], {job_dicts_list[job_id_index]['algorithm_id']: job_id})
        state_training_jobs_mapping.validate()
        state_training_jobs_mappings.append(state_training_jobs_mapping)
    mapping_models = []
    for state_training_job_mapping in state_training_jobs_mappings:
        instance_id = classifier_models.StateTrainingJobsMappingModel.get_entity_id(state_training_job_mapping.exp_id, state_training_job_mapping.exp_version, state_training_job_mapping.state_name)
        mapping_model = classifier_models.StateTrainingJobsMappingModel(id=instance_id, exp_id=state_training_job_mapping.exp_id, exp_version=state_training_job_mapping.exp_version, state_name=state_training_job_mapping.state_name, algorithm_ids_to_job_ids=state_training_job_mapping.algorithm_ids_to_job_ids)
        mapping_models.append(mapping_model)
    models_to_put.extend(mapping_models)
    return models_to_put

def get_new_job_models_for_non_trainable_states(exploration: exp_domain.Exploration, state_names: List[str], exp_versions_diff: exp_domain.ExplorationVersionsDiff) -> Tuple[List[str], List[classifier_models.StateTrainingJobsMappingModel]]:
    if False:
        return 10
    "Returns list of StateTrainingJobsMappingModels for all the\n    state names passed into the function. The mapping is created from the\n    state in the new version of the exploration to the ClassifierTrainingJob of\n    the state in the older version of the exploration. If there's been a change\n    in the state name, we retrieve the old state name and create the mapping\n    accordingly.\n    This method is called only from exp_services._save_exploration() method and\n    is never called from exp_services._create_exploration().\n    In this method, the current_state_name refers to the name of the state in\n    the current version of the exploration whereas the old_state_name refers to\n    the name of the state in the previous version of the exploration.\n    Note that this does not actually create models in the datastore. It just\n    creates instances of the models and returns them. The caller of this method\n    is responsible for the put operation.\n\n    Args:\n        exploration: Exploration. The Exploration domain object.\n        state_names: list(str). List of state names.\n        exp_versions_diff: ExplorationVersionsDiff. An instance of the\n            exploration versions diff class.\n\n    Raises:\n        Exception. This method should not be called by exploration with version\n            number 1.\n\n    Returns:\n        tuple(list(str), list(StateTrainingJobsMappingModel)). A 2-tuple\n        whose elements are as follows:\n        - list(str). State names which don't have classifier model for previous\n            version of exploration.\n        - list(StateTrainingJobsMappingModel). StateTrainingJobsMappingModels\n            for all the state names passed into the function.\n    "
    exp_id = exploration.id
    current_exp_version = exploration.version
    old_exp_version = current_exp_version - 1
    if old_exp_version <= 0:
        raise Exception('This method should not be called by exploration with version number %s' % current_exp_version)
    state_names_to_retrieve = []
    for current_state_name in state_names:
        old_state_name = current_state_name
        if current_state_name in exp_versions_diff.new_to_old_state_names:
            old_state_name = exp_versions_diff.new_to_old_state_names[current_state_name]
        state_names_to_retrieve.append(old_state_name)
    classifier_training_job_maps = get_classifier_training_job_maps(exp_id, old_exp_version, state_names_to_retrieve)
    state_training_jobs_mappings = []
    state_names_without_classifier = []
    for (index, classifier_training_job_map) in enumerate(classifier_training_job_maps):
        if classifier_training_job_map is None:
            logging.error('The ClassifierTrainingJobModel for the %s state of Exploration with exp_id %s and exp_version %s does not exist.' % (state_names_to_retrieve[index], exp_id, old_exp_version))
            state_names_without_classifier.append(state_names_to_retrieve[index])
            continue
        new_state_name = state_names[index]
        algorithm_ids_to_job_ids = dict(classifier_training_job_map.items())
        state_training_jobs_mapping = classifier_domain.StateTrainingJobsMapping(exp_id, current_exp_version, new_state_name, algorithm_ids_to_job_ids)
        state_training_jobs_mapping.validate()
        state_training_jobs_mappings.append(state_training_jobs_mapping)
    mapping_models = []
    for state_training_job_mapping in state_training_jobs_mappings:
        instance_id = classifier_models.StateTrainingJobsMappingModel.get_entity_id(state_training_job_mapping.exp_id, state_training_job_mapping.exp_version, state_training_job_mapping.state_name)
        mapping_model = classifier_models.StateTrainingJobsMappingModel(id=instance_id, exp_id=state_training_job_mapping.exp_id, exp_version=state_training_job_mapping.exp_version, state_name=state_training_job_mapping.state_name, algorithm_ids_to_job_ids=state_training_job_mapping.algorithm_ids_to_job_ids)
        mapping_models.append(mapping_model)
    return (state_names_without_classifier, mapping_models)

def get_classifier_training_job_from_model(classifier_training_job_model: classifier_models.ClassifierTrainingJobModel) -> classifier_domain.ClassifierTrainingJob:
    if False:
        return 10
    'Gets a classifier training job domain object from a classifier\n    training job model.\n\n    Args:\n        classifier_training_job_model: ClassifierTrainingJobModel. Classifier\n            training job instance in datastore.\n\n    Returns:\n        classifier_training_job: ClassifierTrainingJob. Domain object for the\n        classifier training job.\n    '
    return classifier_domain.ClassifierTrainingJob(classifier_training_job_model.id, classifier_training_job_model.algorithm_id, classifier_training_job_model.interaction_id, classifier_training_job_model.exp_id, classifier_training_job_model.exp_version, classifier_training_job_model.next_scheduled_check_time, classifier_training_job_model.state_name, classifier_training_job_model.status, classifier_training_job_model.training_data, classifier_training_job_model.algorithm_version)

def get_classifier_training_job_by_id(job_id: str) -> classifier_domain.ClassifierTrainingJob:
    if False:
        return 10
    'Gets a classifier training job by a job_id.\n\n    Args:\n        job_id: str. ID of the classifier training job.\n\n    Returns:\n        classifier_training_job: ClassifierTrainingJob. Domain object for the\n        classifier training job.\n\n    Raises:\n        Exception. Entity for class ClassifierTrainingJobModel with id not\n            found.\n    '
    classifier_training_job_model = classifier_models.ClassifierTrainingJobModel.get(job_id)
    classifier_training_job = get_classifier_training_job_from_model(classifier_training_job_model)
    return classifier_training_job

def _update_classifier_training_jobs_status(job_ids: List[str], status: str) -> None:
    if False:
        print('Hello World!')
    'Checks for the existence of the model and then updates it.\n\n    Args:\n        job_ids: list(str). List of ID of the ClassifierTrainingJob domain\n            objects.\n        status: str. The status to which the job needs to be updated.\n\n    Raises:\n        Exception. The ClassifierTrainingJobModel corresponding to the job_id\n            of the ClassifierTrainingJob does not exist.\n    '
    classifier_training_job_models_with_none = classifier_models.ClassifierTrainingJobModel.get_multi(job_ids)
    classifier_training_job_models: List[classifier_models.ClassifierTrainingJobModel] = []
    for index in range(len(job_ids)):
        classifier_training_job_model = classifier_training_job_models_with_none[index]
        if classifier_training_job_model is None:
            raise Exception('The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.')
        classifier_training_job = get_classifier_training_job_from_model(classifier_training_job_model)
        classifier_training_job.update_status(status)
        classifier_training_job.validate()
        classifier_training_job_model.status = status
        classifier_training_job_models.append(classifier_training_job_model)
    classifier_models.ClassifierTrainingJobModel.update_timestamps_multi(classifier_training_job_models)
    classifier_models.ClassifierTrainingJobModel.put_multi(classifier_training_job_models)

def mark_training_job_complete(job_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "Updates the training job's status to complete.\n\n    Args:\n        job_id: str. ID of the ClassifierTrainingJob.\n    "
    _update_classifier_training_jobs_status([job_id], feconf.TRAINING_JOB_STATUS_COMPLETE)

def mark_training_jobs_failed(job_ids: List[str]) -> None:
    if False:
        print('Hello World!')
    "Updates the training job's status to failed.\n\n    Args:\n        job_ids: list(str). List of ID of the ClassifierTrainingJobs.\n    "
    _update_classifier_training_jobs_status(job_ids, feconf.TRAINING_JOB_STATUS_FAILED)

def mark_training_job_pending(job_id: str) -> None:
    if False:
        while True:
            i = 10
    "Updates the training job's status to pending.\n\n    Args:\n        job_id: str. ID of the ClassifierTrainingJob.\n    "
    _update_classifier_training_jobs_status([job_id], feconf.TRAINING_JOB_STATUS_PENDING)

def _update_scheduled_check_time_for_new_training_job(job_id: str) -> None:
    if False:
        print('Hello World!')
    'Updates the next scheduled check time of job with status NEW.\n\n    Args:\n        job_id: str. ID of the ClassifierTrainingJob.\n    '
    classifier_training_job_model = classifier_models.ClassifierTrainingJobModel.get(job_id)
    classifier_training_job_model.next_scheduled_check_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=feconf.CLASSIFIER_JOB_TTL_MINS)
    classifier_training_job_model.update_timestamps()
    classifier_training_job_model.put()

def fetch_next_job() -> Optional[classifier_domain.ClassifierTrainingJob]:
    if False:
        i = 10
        return i + 15
    'Gets next job model in the job queue.\n\n    Returns:\n        ClassifierTrainingJob | None. Domain object of the next training Job,\n        and None if no ClassifierTrainingJobModel is found.\n    '
    classifier_training_jobs: Sequence[classifier_models.ClassifierTrainingJobModel] = []
    offset = 0
    valid_jobs: List[classifier_models.ClassifierTrainingJobModel] = []
    timed_out_job_ids = []
    while len(valid_jobs) == 0:
        (classifier_training_jobs, offset) = classifier_models.ClassifierTrainingJobModel.query_new_and_pending_training_jobs(offset)
        if len(classifier_training_jobs) == 0:
            break
        for training_job in classifier_training_jobs:
            if training_job.status == feconf.TRAINING_JOB_STATUS_PENDING:
                if training_job.next_scheduled_check_time <= datetime.datetime.utcnow():
                    timed_out_job_ids.append(training_job.id)
            else:
                valid_jobs.append(training_job)
    if timed_out_job_ids:
        mark_training_jobs_failed(timed_out_job_ids)
    if not valid_jobs:
        return None
    next_job_model = valid_jobs[0]
    next_job = get_classifier_training_job_from_model(next_job_model)
    _update_scheduled_check_time_for_new_training_job(next_job.job_id)
    return next_job

def store_classifier_data(job_id: str, classifier_data_proto: text_classifier_pb2.TextClassifierFrozenModel) -> None:
    if False:
        return 10
    'Checks for the existence of the model and then updates it.\n\n    Args:\n        job_id: str. ID of the ClassifierTrainingJob domain object.\n        classifier_data_proto: FrozenModel. The frozen model protobuf object\n            containing result of training job that needs to be stored.\n\n    Raises:\n        Exception. The ClassifierTrainingJobModel corresponding to the job_id\n            of the ClassifierTrainingJob does not exist.\n    '
    classifier_training_job_model = classifier_models.ClassifierTrainingJobModel.get(job_id, strict=False)
    if not classifier_training_job_model:
        raise Exception('The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.')
    classifier_training_job = get_classifier_training_job_from_model(classifier_training_job_model)
    classifier_training_job.validate()
    fs_services.save_classifier_data(classifier_training_job_model.exp_id, job_id, classifier_data_proto)

def delete_classifier_training_job(job_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes classifier training job model in the datastore given job_id.\n\n    Args:\n        job_id: str. ID of the classifier training job.\n    '
    classifier_training_job_model = classifier_models.ClassifierTrainingJobModel.get(job_id)
    if classifier_training_job_model is not None:
        fs_services.delete_classifier_data(classifier_training_job_model.exp_id, job_id)
        classifier_training_job_model.delete()

def get_classifier_training_job(exp_id: str, exp_version: int, state_name: str, algorithm_id: str) -> Optional[classifier_domain.ClassifierTrainingJob]:
    if False:
        return 10
    'Gets classifier training job object for given algorithm_id for the\n    given <exploration, version, state> triplet.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. The exploration version.\n        state_name: str. The state name for which we retrieve the job.\n        algorithm_id: str. The ID of the algorithm for which classifier training\n            job is to be retrieved.\n\n    Returns:\n        ClassifierTrainingJob|None. An instance for the classifier training job\n        or None if no such instance is found.\n    '
    state_training_jobs_mapping_model = classifier_models.StateTrainingJobsMappingModel.get_model(exp_id, exp_version, state_name)
    if state_training_jobs_mapping_model is None:
        return None
    job_id = state_training_jobs_mapping_model.algorithm_ids_to_job_ids[algorithm_id]
    return get_classifier_training_job_by_id(job_id)

def get_state_training_jobs_mapping(exp_id: str, exp_version: int, state_name: str) -> Optional[classifier_domain.StateTrainingJobsMapping]:
    if False:
        return 10
    'Gets training job exploration mapping model for given exploration state\n    combination.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. Version of the exploration.\n        state_name: str. Name of the state for which training job mapping model\n            is to be retrieved.\n\n    Returns:\n        StateTrainingJobsMapping | None. A domain object containing exploration\n        mapping model information. None, if no such instance is found.\n    '
    state_training_jobs_mapping_model = classifier_models.StateTrainingJobsMappingModel.get_model(exp_id, exp_version, state_name)
    if state_training_jobs_mapping_model is None:
        return None
    return classifier_domain.StateTrainingJobsMapping(state_training_jobs_mapping_model.exp_id, state_training_jobs_mapping_model.exp_version, state_training_jobs_mapping_model.state_name, state_training_jobs_mapping_model.algorithm_ids_to_job_ids)

def migrate_state_training_jobs(state_training_jobs_mapping: classifier_domain.StateTrainingJobsMapping) -> None:
    if False:
        return 10
    "Migrate exploration training jobs to latest version of algorithm_id\n    and algorithm_version.\n\n    This function lazily migrates an older classifier training job and\n    trains new classifiers. Specifically, if training job exploration mapping of\n    an <exploration, version, state> triplet is missing job_id for some\n    algorithm_id, or if the job_id exists but it has been trained on a now\n    obsolete algorithm, we re-submit the jobs.\n\n    The function goes through existing training job exploration mapping and\n    identifies three different types of algorithm IDs.\n        1. algorithm_ids_to_upgrade: Those which exist but needs to be\n            upgraded a new algorithm (because existing one has been deprecated)\n            by re-submitting the training job.\n        2. algorithm_ids_to_add: Those which doesn't exist and needs to be added\n            by submitting a new training job.\n        3. algorithm_ids_to_remove: Those which needs to be removed since these\n            algorithms are no longer supported.\n\n    Once all three types of algorithm IDs are filtered, the function performs\n    specific tasks tailored to each of them. We call this a lazy migration\n    because it happens only when there is a query to retrieve a trained model\n    for given <exploration, version, state> and algorithm_id.\n\n    Args:\n        state_training_jobs_mapping: StateTrainingJobsMapping. Domain\n            object containing exploration to training job id mapping. This\n            mapping is used to figure out jobs that need to be re-submitted,\n            added or removed.\n\n    Raises:\n        Exception. Interaction id does not exist for the state.\n    "
    exp_id = state_training_jobs_mapping.exp_id
    exp_version = state_training_jobs_mapping.exp_version
    state_name = state_training_jobs_mapping.state_name
    exploration = exp_fetchers.get_exploration_by_id(exp_id, version=exp_version)
    interaction_id = exploration.states[state_name].interaction.id
    if interaction_id is None:
        raise Exception('Interaction id does not exist for the state having state_name: %s' % state_name)
    algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id']
    algorithm_version = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_version']
    algorithm_id_to_algorithm_version = {algorithm_id: algorithm_version}
    possible_algorithm_ids = [algorithm_id]
    algorithm_ids_to_add = set(possible_algorithm_ids).difference(set(state_training_jobs_mapping.algorithm_ids_to_job_ids.keys()))
    algorithm_ids_to_remove = set(state_training_jobs_mapping.algorithm_ids_to_job_ids.keys()).difference(set(possible_algorithm_ids))
    algorithm_ids_to_upgrade = set(possible_algorithm_ids).intersection(set(state_training_jobs_mapping.algorithm_ids_to_job_ids.keys()))
    if len(algorithm_ids_to_add) > 0:
        job_dicts: List[JobInfoDict] = []
        for algorithm_id in algorithm_ids_to_add:
            next_scheduled_check_time = datetime.datetime.utcnow()
            training_data = exploration.states[state_name].get_training_data()
            classifier_domain.ClassifierTrainingJob('job_id_dummy', algorithm_id, interaction_id, exp_id, exp_version, next_scheduled_check_time, state_name, feconf.TRAINING_JOB_STATUS_NEW, training_data, algorithm_version).validate()
            job_dicts.append({'algorithm_id': algorithm_id, 'interaction_id': interaction_id, 'exp_id': exp_id, 'exp_version': exp_version, 'next_scheduled_check_time': next_scheduled_check_time, 'state_name': state_name, 'training_data': training_data, 'status': feconf.TRAINING_JOB_STATUS_NEW, 'algorithm_version': algorithm_version})
        job_ids = classifier_models.ClassifierTrainingJobModel.create_multi(job_dicts)
        for (algorithm_id, job_id) in zip(algorithm_ids_to_add, job_ids):
            state_training_jobs_mapping.algorithm_ids_to_job_ids[algorithm_id] = job_id
    if algorithm_ids_to_upgrade:
        for algorithm_id in algorithm_ids_to_upgrade:
            classifier_training_job = classifier_models.ClassifierTrainingJobModel.get_by_id(state_training_jobs_mapping.algorithm_ids_to_job_ids[algorithm_id])
            classifier_training_job.algorithm_version = algorithm_id_to_algorithm_version[algorithm_id]
            classifier_training_job.next_scheduled_check_time = datetime.datetime.utcnow()
            classifier_training_job.status = feconf.TRAINING_JOB_STATUS_NEW
            classifier_training_job.update_timestamps()
            classifier_training_job.put()
    if algorithm_ids_to_remove:
        for algorithm_id in algorithm_ids_to_remove:
            delete_classifier_training_job(state_training_jobs_mapping.algorithm_ids_to_job_ids[algorithm_id])
            state_training_jobs_mapping.algorithm_ids_to_job_ids.pop(algorithm_id)
    state_training_jobs_mapping_model = classifier_models.StateTrainingJobsMappingModel.get_model(exp_id, exp_version, state_name)
    state_training_jobs_mapping.validate()
    assert state_training_jobs_mapping_model is not None
    state_training_jobs_mapping_model.algorithm_ids_to_job_ids = state_training_jobs_mapping.algorithm_ids_to_job_ids
    state_training_jobs_mapping_model.update_timestamps()
    state_training_jobs_mapping_model.put()

def get_classifier_training_job_maps(exp_id: str, exp_version: int, state_names: List[str]) -> List[Optional[Dict[str, str]]]:
    if False:
        print('Hello World!')
    'Gets the list of algorithm-id-to-classifier-training-job mappings for\n    each of the given state names.\n\n    Args:\n        exp_id: str. ID of the exploration.\n        exp_version: int. The exploration version.\n        state_names: list(str). The state names for which we retrieve the job\n            mappings.\n\n    Returns:\n        list(dict(str: str)|None). A list of dicts, each mapping\n        algorithm IDs to the corresponding job IDs. Each element\n        in the list corresponds to the corresponding state name in the\n        state_names input argument. None, if no StateMappingModel exists\n        for corresponding exploration id.\n    '
    state_training_jobs_mapping_models = classifier_models.StateTrainingJobsMappingModel.get_models(exp_id, exp_version, state_names)
    state_to_algorithm_id_job_id_maps: List[Optional[Dict[str, str]]] = []
    for state_mapping_model in state_training_jobs_mapping_models:
        if state_mapping_model:
            algo_ids_to_job_ids: Dict[str, str] = state_mapping_model.algorithm_ids_to_job_ids
            state_to_algorithm_id_job_id_maps.append(algo_ids_to_job_ids)
        else:
            state_to_algorithm_id_job_id_maps.append(None)
    return state_to_algorithm_id_job_id_maps

def create_classifier_training_job_for_reverted_exploration(exploration: exp_domain.Exploration, exploration_to_revert_to: exp_domain.Exploration) -> None:
    if False:
        i = 10
        return i + 15
    'Create classifier training job model when an exploration is reverted.\n\n    Args:\n        exploration: Exploration. Exploration domain object.\n        exploration_to_revert_to: Exploration. Exploration to revert to.\n    '
    classifier_training_job_maps_for_old_version = get_classifier_training_job_maps(exploration.id, exploration_to_revert_to.version, list(exploration_to_revert_to.states.keys()))
    state_training_jobs_mappings = []
    state_names = list(exploration_to_revert_to.states.keys())
    for (index, classifier_training_job_map) in enumerate(classifier_training_job_maps_for_old_version):
        if classifier_training_job_map is not None:
            state_name = state_names[index]
            algorithm_ids_to_job_ids = dict(classifier_training_job_map.items())
            state_training_jobs_mapping = classifier_domain.StateTrainingJobsMapping(exploration.id, exploration.version + 1, state_name, algorithm_ids_to_job_ids)
            state_training_jobs_mapping.validate()
            state_training_jobs_mappings.append(state_training_jobs_mapping)
    classifier_models.StateTrainingJobsMappingModel.create_multi(state_training_jobs_mappings)