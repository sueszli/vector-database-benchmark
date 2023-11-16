"""
Amazon QuickSight List and Get Module.

List and Get MUST be together to avoid circular dependency.
"""
import logging
from typing import Any, Callable, Dict, List, Optional
import boto3
from awswrangler import _utils, exceptions, sts
_logger: logging.Logger = logging.getLogger(__name__)

def _list(func_name: str, attr_name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None, **kwargs: Any) -> List[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    if account_id is None:
        account_id = sts.get_account_id(boto3_session=boto3_session)
    client = _utils.client(service_name='quicksight', session=boto3_session)
    func: Callable[..., Dict[str, Any]] = getattr(client, func_name)
    response: Dict[str, Any] = func(AwsAccountId=account_id, **kwargs)
    next_token: str = response.get('NextToken', None)
    result: List[Dict[str, Any]] = response[attr_name]
    while next_token is not None:
        response = func(AwsAccountId=account_id, NextToken=next_token, **kwargs)
        next_token = response.get('NextToken', None)
        result += response[attr_name]
    return result

def list_dashboards(account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'List dashboards in an AWS account.\n\n    Parameters\n    ----------\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Dashboards.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> dashboards = wr.quicksight.list_dashboards()\n    '
    return _list(func_name='list_dashboards', attr_name='DashboardSummaryList', account_id=account_id, boto3_session=boto3_session)

def list_datasets(account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    'List all QuickSight datasets summaries.\n\n    Parameters\n    ----------\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Datasets summaries.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> datasets = wr.quicksight.list_datasets()\n    '
    return _list(func_name='list_data_sets', attr_name='DataSetSummaries', account_id=account_id, boto3_session=boto3_session)

def list_data_sources(account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'List all QuickSight Data sources summaries.\n\n    Parameters\n    ----------\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Data sources summaries.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> sources = wr.quicksight.list_data_sources()\n    '
    return _list(func_name='list_data_sources', attr_name='DataSources', account_id=account_id, boto3_session=boto3_session)

def list_templates(account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'List all QuickSight templates.\n\n    Parameters\n    ----------\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Templates summaries.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> templates = wr.quicksight.list_templates()\n    '
    return _list(func_name='list_templates', attr_name='TemplateSummaryList', account_id=account_id, boto3_session=boto3_session)

def list_group_memberships(group_name: str, namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'List all QuickSight Group memberships.\n\n    Parameters\n    ----------\n    group_name : str\n        The name of the group that you want to see a membership list of.\n    namespace : str\n        The namespace. Currently, you should set this to default .\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Group memberships.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> memberships = wr.quicksight.list_group_memberships()\n    '
    return _list(func_name='list_group_memberships', attr_name='GroupMemberList', account_id=account_id, boto3_session=boto3_session, GroupName=group_name, Namespace=namespace)

def list_groups(namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'List all QuickSight Groups.\n\n    Parameters\n    ----------\n    namespace : str\n        The namespace. Currently, you should set this to default .\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Groups.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> groups = wr.quicksight.list_groups()\n    '
    return _list(func_name='list_groups', attr_name='GroupList', account_id=account_id, boto3_session=boto3_session, Namespace=namespace)

def list_iam_policy_assignments(status: Optional[str]=None, namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    "List IAM policy assignments in the current Amazon QuickSight account.\n\n    Parameters\n    ----------\n    status : str, optional\n        The status of the assignments.\n        'ENABLED'|'DRAFT'|'DISABLED'\n    namespace : str\n        The namespace. Currently, you should set this to default .\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        IAM policy assignments.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> assigns = wr.quicksight.list_iam_policy_assignments()\n    "
    args: Dict[str, Any] = {'func_name': 'list_iam_policy_assignments', 'attr_name': 'IAMPolicyAssignments', 'account_id': account_id, 'boto3_session': boto3_session, 'Namespace': namespace}
    if status is not None:
        args['AssignmentStatus'] = status
    return _list(**args)

def list_iam_policy_assignments_for_user(user_name: str, namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'List all the IAM policy assignments.\n\n    Including the Amazon Resource Names (ARNs) for the IAM policies assigned\n    to the specified user and group or groups that the user belongs to.\n\n    Parameters\n    ----------\n    user_name : str\n        The name of the user.\n    namespace : str\n        The namespace. Currently, you should set this to default .\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        IAM policy assignments.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> assigns = wr.quicksight.list_iam_policy_assignments_for_user()\n    '
    return _list(func_name='list_iam_policy_assignments_for_user', attr_name='ActiveAssignments', account_id=account_id, boto3_session=boto3_session, UserName=user_name, Namespace=namespace)

def list_user_groups(user_name: str, namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'List the Amazon QuickSight groups that an Amazon QuickSight user is a member of.\n\n    Parameters\n    ----------\n    user_name: str:\n        The Amazon QuickSight user name that you want to list group memberships for.\n    namespace : str\n        The namespace. Currently, you should set this to default .\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Groups.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> groups = wr.quicksight.list_user_groups()\n    '
    return _list(func_name='list_user_groups', attr_name='GroupList', account_id=account_id, boto3_session=boto3_session, UserName=user_name, Namespace=namespace)

def list_users(namespace: str='default', account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'Return a list of all of the Amazon QuickSight users belonging to this account.\n\n    Parameters\n    ----------\n    namespace : str\n        The namespace. Currently, you should set this to default.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        Groups.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> users = wr.quicksight.list_users()\n    '
    return _list(func_name='list_users', attr_name='UserList', account_id=account_id, boto3_session=boto3_session, Namespace=namespace)

def list_ingestions(dataset_name: Optional[str]=None, dataset_id: Optional[str]=None, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'List the history of SPICE ingestions for a dataset.\n\n    Parameters\n    ----------\n    dataset_name : str, optional\n        Dataset name.\n    dataset_id : str, optional\n        The ID of the dataset used in the ingestion.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]]\n        IAM policy assignments.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> ingestions = wr.quicksight.list_ingestions()\n    '
    if dataset_name is None and dataset_id is None:
        raise exceptions.InvalidArgument('You must pass a not None name or dataset_id argument.')
    if account_id is None:
        account_id = sts.get_account_id(boto3_session=boto3_session)
    if dataset_id is None and dataset_name is not None:
        dataset_id = get_dataset_id(name=dataset_name, account_id=account_id, boto3_session=boto3_session)
    return _list(func_name='list_ingestions', attr_name='Ingestions', account_id=account_id, boto3_session=boto3_session, DataSetId=dataset_id)

def _get_ids(name: str, func: Callable[..., List[Dict[str, Any]]], attr_name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    ids: List[str] = []
    for item in func(account_id=account_id, boto3_session=boto3_session):
        if item['Name'] == name:
            ids.append(item[attr_name])
    return ids

def _get_id(name: str, func: Callable[..., List[Dict[str, Any]]], attr_name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        print('Hello World!')
    ids: List[str] = _get_ids(name=name, func=func, attr_name=attr_name, account_id=account_id, boto3_session=boto3_session)
    if len(ids) == 0:
        raise exceptions.InvalidArgument(f'There is no {attr_name} related with name {name}')
    if len(ids) > 1:
        raise exceptions.InvalidArgument(f'There is {len(ids)} {attr_name} with name {name}. Please pass the id argument to specify which one you would like to describe.')
    return ids[0]

def get_dashboard_ids(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get QuickSight dashboard IDs given a name.\n\n    Note\n    ----\n    This function returns a list of ID because Quicksight accepts duplicated dashboard names,\n    so you may have more than 1 ID for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Dashboard name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[str]\n        Dashboard IDs.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> ids = wr.quicksight.get_dashboard_ids(name="...")\n    '
    return _get_ids(name=name, func=list_dashboards, attr_name='DashboardId', account_id=account_id, boto3_session=boto3_session)

def get_dashboard_id(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        while True:
            i = 10
    'Get QuickSight dashboard ID given a name and fails if there is more than 1 ID associated with this name.\n\n    Parameters\n    ----------\n    name : str\n        Dashboard name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Dashboard ID.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> my_id = wr.quicksight.get_dashboard_id(name="...")\n    '
    return _get_id(name=name, func=list_dashboards, attr_name='DashboardId', account_id=account_id, boto3_session=boto3_session)

def get_dataset_ids(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get QuickSight dataset IDs given a name.\n\n    Note\n    ----\n    This function returns a list of ID because Quicksight accepts duplicated datasets names,\n    so you may have more than 1 ID for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Dataset name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[str]\n        Datasets IDs.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> ids = wr.quicksight.get_dataset_ids(name="...")\n    '
    return _get_ids(name=name, func=list_datasets, attr_name='DataSetId', account_id=account_id, boto3_session=boto3_session)

def get_dataset_id(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        while True:
            i = 10
    'Get QuickSight Dataset ID given a name and fails if there is more than 1 ID associated with this name.\n\n    Parameters\n    ----------\n    name : str\n        Dataset name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Dataset ID.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> my_id = wr.quicksight.get_dataset_id(name="...")\n    '
    return _get_id(name=name, func=list_datasets, attr_name='DataSetId', account_id=account_id, boto3_session=boto3_session)

def get_data_source_ids(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Get QuickSight data source IDs given a name.\n\n    Note\n    ----\n    This function returns a list of ID because Quicksight accepts duplicated data source names,\n    so you may have more than 1 ID for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Data source name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[str]\n        Data source IDs.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> ids = wr.quicksight.get_data_source_ids(name="...")\n    '
    return _get_ids(name=name, func=list_data_sources, attr_name='DataSourceId', account_id=account_id, boto3_session=boto3_session)

def get_data_source_id(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        return 10
    'Get QuickSight data source ID given a name and fails if there is more than 1 ID associated with this name.\n\n    Parameters\n    ----------\n    name : str\n        Data source name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Dataset ID.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> my_id = wr.quicksight.get_data_source_id(name="...")\n    '
    return _get_id(name=name, func=list_data_sources, attr_name='DataSourceId', account_id=account_id, boto3_session=boto3_session)

def get_template_ids(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Get QuickSight template IDs given a name.\n\n    Note\n    ----\n    This function returns a list of ID because Quicksight accepts duplicated templates names,\n    so you may have more than 1 ID for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Template name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[str]\n        Template IDs.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> ids = wr.quicksight.get_template_ids(name="...")\n    '
    return _get_ids(name=name, func=list_templates, attr_name='TemplateId', account_id=account_id, boto3_session=boto3_session)

def get_template_id(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        print('Hello World!')
    'Get QuickSight template ID given a name and fails if there is more than 1 ID associated with this name.\n\n    Parameters\n    ----------\n    name : str\n        Template name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Template ID.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> my_id = wr.quicksight.get_template_id(name="...")\n    '
    return _get_id(name=name, func=list_templates, attr_name='TemplateId', account_id=account_id, boto3_session=boto3_session)

def get_data_source_arns(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Get QuickSight Data source ARNs given a name.\n\n    Note\n    ----\n    This function returns a list of ARNs because Quicksight accepts duplicated data source names,\n    so you may have more than 1 ARN for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Data source name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[str]\n        Data source ARNs.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> arns = wr.quicksight.get_data_source_arns(name="...")\n    '
    arns: List[str] = []
    for source in list_data_sources(account_id=account_id, boto3_session=boto3_session):
        if source['Name'] == name:
            arns.append(source['Arn'])
    return arns

def get_data_source_arn(name: str, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get QuickSight data source ARN given a name and fails if there is more than 1 ARN associated with this name.\n\n    Note\n    ----\n    This function returns a list of ARNs because Quicksight accepts duplicated data source names,\n    so you may have more than 1 ARN for a given name.\n\n    Parameters\n    ----------\n    name : str\n        Data source name.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Data source ARN.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> arn = wr.quicksight.get_data_source_arn("...")\n    '
    arns: List[str] = get_data_source_arns(name=name, account_id=account_id, boto3_session=boto3_session)
    if len(arns) == 0:
        raise exceptions.InvalidArgument(f'There is not data source with name {name}')
    if len(arns) > 1:
        raise exceptions.InvalidArgument(f'There is more than 1 data source with name {name}. Please pass the data_source_arn argument to specify which one you would like to describe.')
    return arns[0]