"""
Support for the Amazon Simple Queue Service.
"""
import logging
import salt.utils.json
import salt.utils.path
log = logging.getLogger(__name__)
_OUTPUT = '--output json'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if salt.utils.path.which('aws'):
        return True
    return (False, 'The module aws_sqs could not be loaded: aws command not found')

def _region(region):
    if False:
        return 10
    '\n    Return the region argument.\n    '
    return ' --region {r}'.format(r=region)

def _run_aws(cmd, region, opts, user, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Runs the given command against AWS.\n    cmd\n        Command to run\n    region\n        Region to execute cmd in\n    opts\n        Pass in from salt\n    user\n        Pass in from salt\n    kwargs\n        Key-value arguments to pass to the command\n    '
    receipthandle = kwargs.pop('receipthandle', None)
    if receipthandle:
        kwargs['receipt-handle'] = receipthandle
    num = kwargs.pop('num', None)
    if num:
        kwargs['max-number-of-messages'] = num
    _formatted_args = ['--{} "{}"'.format(k, v) for (k, v) in kwargs.items()]
    cmd = 'aws sqs {cmd} {args} {region} {out}'.format(cmd=cmd, args=' '.join(_formatted_args), region=_region(region), out=_OUTPUT)
    rtn = __salt__['cmd.run'](cmd, runas=user, python_shell=False)
    return salt.utils.json.loads(rtn) if rtn else ''

def receive_message(queue, region, num=1, opts=None, user=None):
    if False:
        i = 10
        return i + 15
    "\n    Receive one or more messages from a queue in a region\n\n    queue\n        The name of the queue to receive messages from\n\n    region\n        Region where SQS queues exists\n\n    num : 1\n        The max number of messages to receive\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.receive_message <sqs queue> <region>\n        salt '*' aws_sqs.receive_message <sqs queue> <region> num=10\n\n    .. versionadded:: 2014.7.0\n\n    "
    ret = {'Messages': None}
    queues = list_queues(region, opts, user)
    url_map = _parse_queue_list(queues)
    if queue not in url_map:
        log.info('"%s" queue does not exist.', queue)
        return ret
    out = _run_aws('receive-message', region, opts, user, queue=url_map[queue], num=num)
    ret['Messages'] = out['Messages']
    return ret

def delete_message(queue, region, receipthandle, opts=None, user=None):
    if False:
        print('Hello World!')
    "\n    Delete one or more messages from a queue in a region\n\n    queue\n        The name of the queue to delete messages from\n\n    region\n        Region where SQS queues exists\n\n    receipthandle\n        The ReceiptHandle of the message to delete. The ReceiptHandle\n        is obtained in the return from receive_message\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.delete_message <sqs queue> <region> receipthandle='<sqs ReceiptHandle>'\n\n    .. versionadded:: 2014.7.0\n\n    "
    queues = list_queues(region, opts, user)
    url_map = _parse_queue_list(queues)
    if queue not in url_map:
        log.info('"%s" queue does not exist.', queue)
        return False
    out = _run_aws('delete-message', region, opts, user, receipthandle=receipthandle, queue=url_map[queue])
    return True

def list_queues(region, opts=None, user=None):
    if False:
        while True:
            i = 10
    "\n    List the queues in the selected region.\n\n    region\n        Region to list SQS queues for\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.list_queues <region>\n\n    "
    out = _run_aws('list-queues', region, opts, user)
    ret = {'retcode': 0, 'stdout': out['QueueUrls']}
    return ret

def create_queue(name, region, opts=None, user=None):
    if False:
        i = 10
        return i + 15
    "\n    Creates a queue with the correct name.\n\n    name\n        Name of the SQS queue to create\n\n    region\n        Region to create the SQS queue in\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.create_queue <sqs queue> <region>\n\n    "
    create = {'queue-name': name}
    out = _run_aws('create-queue', region=region, opts=opts, user=user, **create)
    ret = {'retcode': 0, 'stdout': out['QueueUrl'], 'stderr': ''}
    return ret

def delete_queue(name, region, opts=None, user=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Deletes a queue in the region.\n\n    name\n        Name of the SQS queue to deletes\n    region\n        Name of the region to delete the queue from\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.delete_queue <sqs queue> <region>\n\n    "
    queues = list_queues(region, opts, user)
    url_map = _parse_queue_list(queues)
    log.debug('map %s', url_map)
    if name in url_map:
        delete = {'queue-url': url_map[name]}
        rtn = _run_aws('delete-queue', region=region, opts=opts, user=user, **delete)
        success = True
        err = ''
        out = '{} deleted'.format(name)
    else:
        out = ''
        err = 'Delete failed'
        success = False
    ret = {'retcode': 0 if success else 1, 'stdout': out, 'stderr': err}
    return ret

def queue_exists(name, region, opts=None, user=None):
    if False:
        return 10
    "\n    Returns True or False on whether the queue exists in the region\n\n    name\n        Name of the SQS queue to search for\n\n    region\n        Name of the region to search for the queue in\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' aws_sqs.queue_exists <sqs queue> <region>\n\n    "
    output = list_queues(region, opts, user)
    return name in _parse_queue_list(output)

def _parse_queue_list(list_output):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse the queue to get a dict of name -> URL\n    '
    queues = {q.split('/')[-1]: q for q in list_output['stdout']}
    return queues