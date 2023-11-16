"""
.. module: security_monkey.common.sts_connect
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.datastore import Account
import botocore.session
import boto3
import boto
from security_monkey import app, AWS_DEFAULT_REGION, ARN_PREFIX

def connect(account_name, connection_type, **args):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Examples of use:\n    ec2 = sts_connect.connect(environment, 'ec2', region=region, validate_certs=False)\n    ec2 = sts_connect.connect(environment, 'ec2', validate_certs=False, debug=1000)\n    ec2 = sts_connect.connect(environment, 'ec2')\n    where environment is ( test, prod, dev )\n    s3  = sts_connect.connect(environment, 's3')\n    ses = sts_connect.connect(environment, 'ses')\n\n    :param account: Account to connect with (i.e. test, prod, dev)\n\n    :raises Exception: RDS Region not valid\n                       AWS Tech not supported.\n\n    :returns: STS Connection Object for given tech\n\n    :note: To use this method a SecurityMonkey role must be created\n            in the target account with full read only privileges.\n    "
    region = AWS_DEFAULT_REGION
    if 'assumed_role' in args:
        role = args['assumed_role']
    else:
        account = Account.query.filter(Account.name == account_name).first()
        sts = boto3.client('sts', region_name=region)
        role_name = 'SecurityMonkey'
        external_id = None
        if account.getCustom('role_name') and account.getCustom('role_name') != '':
            role_name = account.getCustom('role_name')
        if account.getCustom('external_id') and account.getCustom('external_id') != '':
            external_id = account.getCustom('external_id')
        arn = ARN_PREFIX + ':iam::' + account.identifier + ':role/' + role_name
        assume_role_kwargs = {'RoleArn': arn, 'RoleSessionName': 'secmonkey'}
        if external_id:
            assume_role_kwargs['ExternalId'] = external_id
        role = sts.assume_role(**assume_role_kwargs)
    if connection_type == 'botocore':
        botocore_session = botocore.session.get_session()
        botocore_session.set_credentials(role['Credentials']['AccessKeyId'], role['Credentials']['SecretAccessKey'], token=role['Credentials']['SessionToken'])
        return botocore_session
    if 'region' in args:
        region = args.pop('region')
        if hasattr(region, 'name'):
            region = region.name
    if 'boto3' in connection_type:
        (_, tech, api) = connection_type.split('.')
        session = boto3.Session(aws_access_key_id=role['Credentials']['AccessKeyId'], aws_secret_access_key=role['Credentials']['SecretAccessKey'], aws_session_token=role['Credentials']['SessionToken'], region_name=region)
        if api == 'resource':
            return session.resource(tech)
        return session.client(tech)
    module = __import__('boto.{}'.format(connection_type))
    for subm in connection_type.split('.'):
        module = getattr(module, subm)
    return module.connect_to_region(region, aws_access_key_id=role['Credentials']['AccessKeyId'], aws_secret_access_key=role['Credentials']['SecretAccessKey'], security_token=role['Credentials']['SessionToken'])