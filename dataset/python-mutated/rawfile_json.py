"""
Take data from salt and "return" it into a raw file containing the json, with
one line per event.

Add the following to the minion or master configuration file.

.. code-block:: yaml

    rawfile_json.filename: <path_to_output_file>

Default is ``/var/log/salt/events``.

Common use is to log all events on the master. This can generate a lot of
noise, so you may wish to configure batch processing and/or configure the
:conf_master:`event_return_whitelist` or :conf_master:`event_return_blacklist`
to restrict the events that are written.
"""
import logging
import salt.returners
import salt.utils.files
import salt.utils.json
log = logging.getLogger(__name__)
__virtualname__ = 'rawfile_json'

def __virtual__():
    if False:
        i = 10
        return i + 15
    return __virtualname__

def _get_options(ret):
    if False:
        return 10
    '\n    Returns options used for the rawfile_json returner.\n    '
    defaults = {'filename': '/var/log/salt/events'}
    attrs = {'filename': 'filename'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__, defaults=defaults)
    return _options

def returner(ret):
    if False:
        while True:
            i = 10
    '\n    Write the return data to a file on the minion.\n    '
    opts = _get_options(ret)
    try:
        with salt.utils.files.flopen(opts['filename'], 'a') as logfile:
            salt.utils.json.dump(ret, logfile)
            logfile.write('\n')
    except Exception:
        log.error('Could not write to rawdata_json file %s', opts['filename'])
        raise

def event_return(events):
    if False:
        return 10
    '\n    Write event data (return data and non-return data) to file on the master.\n    '
    if len(events) == 0:
        return
    opts = _get_options({})
    try:
        with salt.utils.files.flopen(opts['filename'], 'a') as logfile:
            for event in events:
                salt.utils.json.dump(event, logfile)
                logfile.write('\n')
    except Exception:
        log.error('Could not write to rawdata_json file %s', opts['filename'])
        raise