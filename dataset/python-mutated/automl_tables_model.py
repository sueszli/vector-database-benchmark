"""This application demonstrates how to perform basic operations on model
with the Google AutoML Tables API.

For more information, the documentation at
https://cloud.google.com/automl-tables/docs.
"""
import argparse
import os

def create_model(project_id, compute_region, dataset_display_name, model_display_name, train_budget_milli_node_hours, include_column_spec_names=None, exclude_column_spec_names=None):
    if False:
        while True:
            i = 10
    'Create a model.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.create_model(model_display_name, train_budget_milli_node_hours=train_budget_milli_node_hours, dataset_display_name=dataset_display_name, include_column_spec_names=include_column_spec_names, exclude_column_spec_names=exclude_column_spec_names)
    print('Training model...')
    print(f'Training operation name: {response.operation.name}')
    print(f'Training completed: {response.result()}')

def get_operation_status(operation_full_id):
    if False:
        print('Hello World!')
    'Get operation status.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient()
    op = client.auto_ml_client._transport.operations_client.get_operation(operation_full_id)
    print(f'Operation status: {op}')

def list_models(project_id, compute_region, filter=None):
    if False:
        print('Hello World!')
    'List all models.'
    result = []
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.list_models(filter=filter)
    print('List of models:')
    for model in response:
        if model.deployment_state == automl.Model.DeploymentState.DEPLOYED:
            deployment_state = 'deployed'
        else:
            deployment_state = 'undeployed'
        print(f'Model name: {model.name}')
        print('Model id: {}'.format(model.name.split('/')[-1]))
        print(f'Model display name: {model.display_name}')
        metadata = model.tables_model_metadata
        print('Target column display name: {}'.format(metadata.target_column_spec.display_name))
        print('Training budget in node milli hours: {}'.format(metadata.train_budget_milli_node_hours))
        print('Training cost in node milli hours: {}'.format(metadata.train_cost_milli_node_hours))
        print(f'Model create time: {model.create_time}')
        print(f'Model deployment state: {deployment_state}')
        print('\n')
        result.append(model)
    return result

def get_model(project_id, compute_region, model_display_name):
    if False:
        print('Hello World!')
    'Get model details.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    model = client.get_model(model_display_name=model_display_name)
    if model.deployment_state == automl.Model.DeploymentState.DEPLOYED:
        deployment_state = 'deployed'
    else:
        deployment_state = 'undeployed'
    feat_list = [(column.feature_importance, column.column_display_name) for column in model.tables_model_metadata.tables_model_column_info]
    feat_list.sort(reverse=True)
    if len(feat_list) < 10:
        feat_to_show = len(feat_list)
    else:
        feat_to_show = 10
    print(f'Model name: {model.name}')
    print('Model id: {}'.format(model.name.split('/')[-1]))
    print(f'Model display name: {model.display_name}')
    print('Features of top importance:')
    for feat in feat_list[:feat_to_show]:
        print(feat)
    print(f'Model create time: {model.create_time}')
    print(f'Model deployment state: {deployment_state}')
    return model

def list_model_evaluations(project_id, compute_region, model_display_name, filter=None):
    if False:
        for i in range(10):
            print('nop')
    'List model evaluations.'
    result = []
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.list_model_evaluations(model_display_name=model_display_name, filter=filter)
    print('List of model evaluations:')
    for evaluation in response:
        print(f'Model evaluation name: {evaluation.name}')
        print('Model evaluation id: {}'.format(evaluation.name.split('/')[-1]))
        print('Model evaluation example count: {}'.format(evaluation.evaluated_example_count))
        print(f'Model evaluation time: {evaluation.create_time}')
        print('\n')
        result.append(evaluation)
    return result

def get_model_evaluation(project_id, compute_region, model_id, model_evaluation_id):
    if False:
        i = 10
        return i + 15
    'Get model evaluation.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient()
    model_path = client.auto_ml_client.model_path(project_id, compute_region, model_id)
    model_evaluation_full_id = f'{model_path}/modelEvaluations/{model_evaluation_id}'
    response = client.get_model_evaluation(model_evaluation_name=model_evaluation_full_id)
    print(response)
    return response

def display_evaluation(project_id, compute_region, model_display_name, filter=None):
    if False:
        for i in range(10):
            print('nop')
    'Display evaluation.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.list_model_evaluations(model_display_name=model_display_name, filter=filter)
    for evaluation in response:
        if not evaluation.annotation_spec_id:
            model_evaluation_name = evaluation.name
            break
    model_evaluation = client.get_model_evaluation(model_evaluation_name=model_evaluation_name)
    classification_metrics = model_evaluation.classification_evaluation_metrics
    if str(classification_metrics):
        confidence_metrics = classification_metrics.confidence_metrics_entry
        print('Model classification metrics (threshold at 0.5):')
        for confidence_metrics_entry in confidence_metrics:
            if confidence_metrics_entry.confidence_threshold == 0.5:
                print('Model Precision: {}%'.format(round(confidence_metrics_entry.precision * 100, 2)))
                print('Model Recall: {}%'.format(round(confidence_metrics_entry.recall * 100, 2)))
                print('Model F1 score: {}%'.format(round(confidence_metrics_entry.f1_score * 100, 2)))
        print(f'Model AUPRC: {classification_metrics.au_prc}')
        print(f'Model AUROC: {classification_metrics.au_roc}')
        print(f'Model log loss: {classification_metrics.log_loss}')
    regression_metrics = model_evaluation.regression_evaluation_metrics
    if str(regression_metrics):
        print('Model regression metrics:')
        print(f'Model RMSE: {regression_metrics.root_mean_squared_error}')
        print(f'Model MAE: {regression_metrics.mean_absolute_error}')
        print('Model MAPE: {}'.format(regression_metrics.mean_absolute_percentage_error))
        print(f'Model R^2: {regression_metrics.r_squared}')

def deploy_model(project_id, compute_region, model_display_name):
    if False:
        while True:
            i = 10
    'Deploy model.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.deploy_model(model_display_name=model_display_name)
    print(f'Model deployed. {response.result()}')

def undeploy_model(project_id, compute_region, model_display_name):
    if False:
        i = 10
        return i + 15
    'Undeploy model.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.undeploy_model(model_display_name=model_display_name)
    print(f'Model undeployed. {response.result()}')

def delete_model(project_id, compute_region, model_display_name):
    if False:
        for i in range(10):
            print('nop')
    'Delete a model.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.delete_model(model_display_name=model_display_name)
    print(f'Model deleted. {response.result()}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    create_model_parser = subparsers.add_parser('create_model', help=create_model.__doc__)
    create_model_parser.add_argument('--dataset_display_name')
    create_model_parser.add_argument('--model_display_name')
    create_model_parser.add_argument('--train_budget_milli_node_hours', type=int)
    get_operation_status_parser = subparsers.add_parser('get_operation_status', help=get_operation_status.__doc__)
    get_operation_status_parser.add_argument('--operation_full_id')
    list_models_parser = subparsers.add_parser('list_models', help=list_models.__doc__)
    list_models_parser.add_argument('--filter_')
    get_model_parser = subparsers.add_parser('get_model', help=get_model.__doc__)
    get_model_parser.add_argument('--model_display_name')
    list_model_evaluations_parser = subparsers.add_parser('list_model_evaluations', help=list_model_evaluations.__doc__)
    list_model_evaluations_parser.add_argument('--model_display_name')
    list_model_evaluations_parser.add_argument('--filter_')
    get_model_evaluation_parser = subparsers.add_parser('get_model_evaluation', help=get_model_evaluation.__doc__)
    get_model_evaluation_parser.add_argument('--model_id')
    get_model_evaluation_parser.add_argument('--model_evaluation_id')
    display_evaluation_parser = subparsers.add_parser('display_evaluation', help=display_evaluation.__doc__)
    display_evaluation_parser.add_argument('--model_display_name')
    display_evaluation_parser.add_argument('--filter_')
    deploy_model_parser = subparsers.add_parser('deploy_model', help=deploy_model.__doc__)
    deploy_model_parser.add_argument('--model_display_name')
    undeploy_model_parser = subparsers.add_parser('undeploy_model', help=undeploy_model.__doc__)
    undeploy_model_parser.add_argument('--model_display_name')
    delete_model_parser = subparsers.add_parser('delete_model', help=delete_model.__doc__)
    delete_model_parser.add_argument('--model_display_name')
    project_id = os.environ['PROJECT_ID']
    compute_region = os.environ['REGION_NAME']
    args = parser.parse_args()
    if args.command == 'create_model':
        create_model(project_id, compute_region, args.dataset_display_name, args.model_display_name, args.train_budget_milli_node_hours)
    if args.command == 'get_operation_status':
        get_operation_status(args.operation_full_id)
    if args.command == 'list_models':
        list_models(project_id, compute_region, args.filter_)
    if args.command == 'get_model':
        get_model(project_id, compute_region, args.model_display_name)
    if args.command == 'list_model_evaluations':
        list_model_evaluations(project_id, compute_region, args.model_display_name, args.filter_)
    if args.command == 'get_model_evaluation':
        get_model_evaluation(project_id, compute_region, args.model_display_name, args.model_evaluation_id)
    if args.command == 'display_evaluation':
        display_evaluation(project_id, compute_region, args.model_display_name, args.filter_)
    if args.command == 'deploy_model':
        deploy_model(project_id, compute_region, args.model_display_name)
    if args.command == 'undeploy_model':
        undeploy_model(project_id, compute_region, args.model_display_name)
    if args.command == 'delete_model':
        delete_model(project_id, compute_region, args.model_display_name)