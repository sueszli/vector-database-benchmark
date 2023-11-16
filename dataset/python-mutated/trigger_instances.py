"""
Module with utility functions for purging old trigger instance objects.
"""
from __future__ import absolute_import
import six
from mongoengine.errors import InvalidQueryError
from st2common.persistence.trigger import TriggerInstance
from st2common.util import isotime
__all__ = ['purge_trigger_instances']

def purge_trigger_instances(logger, timestamp):
    if False:
        return 10
    '\n    :param timestamp: Trigger instances older than this timestamp will be deleted.\n    :type timestamp: ``datetime.datetime\n    '
    if not timestamp:
        raise ValueError('Specify a valid timestamp to purge.')
    logger.info('Purging trigger instances older than timestamp: %s' % timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
    query_filters = {'occurrence_time__lt': isotime.parse(timestamp)}
    try:
        deleted_count = TriggerInstance.delete_by_query(**query_filters)
    except InvalidQueryError as e:
        msg = 'Bad query (%s) used to delete trigger instances: %sPlease contact support.' % (query_filters, six.text_type(e))
        raise InvalidQueryError(msg)
    except:
        logger.exception('Deleting instances using query_filters %s failed.', query_filters)
    else:
        logger.info('Deleted %s trigger instance objects' % deleted_count)
    logger.info('All trigger instance models older than timestamp %s were deleted.', timestamp)