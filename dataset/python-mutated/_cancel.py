"""Amazon QuickSight Cancel Module."""
import logging
from typing import Optional, cast
import boto3
from awswrangler import _utils, exceptions, sts
from awswrangler.quicksight._get_list import get_dataset_id
_logger: logging.Logger = logging.getLogger(__name__)

def cancel_ingestion(ingestion_id: str, dataset_name: Optional[str]=None, dataset_id: Optional[str]=None, account_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> None:
    if False:
        while True:
            i = 10
    'Cancel an ongoing ingestion of data into SPICE.\n\n    Note\n    ----\n    You must pass a not None value for ``dataset_name`` or ``dataset_id`` argument.\n\n    Parameters\n    ----------\n    ingestion_id : str\n        Ingestion ID.\n    dataset_name : str, optional\n        Dataset name.\n    dataset_id : str, optional\n        Dataset ID.\n    account_id : str, optional\n        If None, the account ID will be inferred from your boto3 session.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>>  wr.quicksight.cancel_ingestion(ingestion_id="...", dataset_name="...")\n    '
    if dataset_name is None and dataset_id is None:
        raise exceptions.InvalidArgument('You must pass a not None name or dataset_id argument.')
    if account_id is None:
        account_id = sts.get_account_id(boto3_session=boto3_session)
    if dataset_id is None and dataset_name is not None:
        dataset_id = get_dataset_id(name=dataset_name, account_id=account_id, boto3_session=boto3_session)
    client = _utils.client(service_name='quicksight', session=boto3_session)
    dataset_id = cast(str, dataset_id)
    client.cancel_ingestion(IngestionId=ingestion_id, AwsAccountId=account_id, DataSetId=dataset_id)