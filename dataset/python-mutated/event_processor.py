import logging
import threading
import salt.netapi
import salt.utils.json
log = logging.getLogger(__name__)

class SaltInfo:
    """
    Class to  handle processing and publishing of "real time" Salt upates.
    """

    def __init__(self, handler):
        if False:
            for i in range(10):
                print('nop')
        '\n        handler is expected to be the server side end of a websocket\n        connection.\n        '
        self.handler = handler
        self.jobs = {}
        self.minions = {}

    def publish_minions(self):
        if False:
            i = 10
            return i + 15
        '\n        Publishes minions as a list of dicts.\n        '
        log.debug('in publish minions')
        minions = {}
        log.debug('starting loop')
        for (minion, minion_info) in self.minions.items():
            log.debug(minion)
            curr_minion = {}
            curr_minion.update(minion_info)
            curr_minion.update({'id': minion})
            minions[minion] = curr_minion
        log.debug('ended loop')
        ret = {'minions': minions}
        self.handler.write_message(salt.utils.json.dumps(ret) + '\n\n')

    def publish(self, key, data):
        if False:
            return 10
        '\n        Publishes the data to the event stream.\n        '
        publish_data = {key: data}
        pub = salt.utils.json.dumps(publish_data) + '\n\n'
        self.handler.write_message(pub)

    def process_minion_update(self, event_data):
        if False:
            return 10
        '\n        Associate grains data with a minion and publish minion update\n        '
        tag = event_data['tag']
        event_info = event_data['data']
        mid = tag.split('/')[-1]
        if not self.minions.get(mid, None):
            self.minions[mid] = {}
        minion = self.minions[mid]
        minion.update({'grains': event_info['return']})
        log.debug('In process minion grains update with minions=%s', self.minions)
        self.publish_minions()

    def process_ret_job_event(self, event_data):
        if False:
            i = 10
            return i + 15
        '\n        Process a /ret event returned by Salt for a particular minion.\n        These events contain the returned results from a particular execution.\n        '
        tag = event_data['tag']
        event_info = event_data['data']
        (_, _, jid, _, mid) = tag.split('/')
        job = self.jobs.setdefault(jid, {})
        minion = job.setdefault('minions', {}).setdefault(mid, {})
        minion.update({'return': event_info['return']})
        minion.update({'retcode': event_info['retcode']})
        minion.update({'success': event_info['success']})
        job_complete = all([minion['success'] for (mid, minion) in job['minions'].items()])
        if job_complete:
            job['state'] = 'complete'
        self.publish('jobs', self.jobs)

    def process_new_job_event(self, event_data):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new job with properties from the event data\n        like jid, function, args, timestamp.\n\n        Also sets the initial state to started.\n\n        Minions that are participating in this job are also noted.\n\n        '
        job = None
        tag = event_data['tag']
        event_info = event_data['data']
        minions = {}
        for mid in event_info['minions']:
            minions[mid] = {'success': False}
        job = {'jid': event_info['jid'], 'start_time': event_info['_stamp'], 'minions': minions, 'fun': event_info['fun'], 'tgt': event_info['tgt'], 'tgt_type': event_info['tgt_type'], 'state': 'running'}
        self.jobs[event_info['jid']] = job
        self.publish('jobs', self.jobs)

    def process_key_event(self, event_data):
        if False:
            while True:
                i = 10
        "\n        Tag: salt/key\n        Data:\n        {'_stamp': '2014-05-20T22:45:04.345583',\n         'act': 'delete',\n         'id': 'compute.home',\n         'result': True}\n        "
        tag = event_data['tag']
        event_info = event_data['data']
        if event_info['act'] == 'delete':
            self.minions.pop(event_info['id'], None)
        elif event_info['act'] == 'accept':
            self.minions.setdefault(event_info['id'], {})
        self.publish_minions()

    def process_presence_events(self, salt_data, token, opts):
        if False:
            i = 10
            return i + 15
        '\n        Check if any minions have connected or dropped.\n        Send a message to the client if they have.\n        '
        log.debug('In presence')
        changed = False
        if set(salt_data['data'].get('lost', [])):
            dropped_minions = set(salt_data['data'].get('lost', []))
        else:
            dropped_minions = set(self.minions) - set(salt_data['data'].get('present', []))
        for minion in dropped_minions:
            changed = True
            log.debug('Popping %s', minion)
            self.minions.pop(minion, None)
        if set(salt_data['data'].get('new', [])):
            log.debug('got new minions')
            new_minions = set(salt_data['data'].get('new', []))
            changed = True
        elif set(salt_data['data'].get('present', [])) - set(self.minions):
            log.debug('detected new minions')
            new_minions = set(salt_data['data'].get('present', [])) - set(self.minions)
            changed = True
        else:
            new_minions = []
        tgt = ','.join(new_minions)
        for mid in new_minions:
            log.debug('Adding minion')
            self.minions[mid] = {}
        if tgt:
            changed = True
            client = salt.netapi.NetapiClient(opts)
            client.run({'fun': 'grains.items', 'tgt': tgt, 'expr_type': 'list', 'mode': 'client', 'client': 'local', 'asynchronous': 'local_async', 'token': token})
        if changed:
            self.publish_minions()

    def process(self, salt_data, token, opts):
        if False:
            i = 10
            return i + 15
        '\n        Process events and publish data\n        '
        log.debug('In process %s', threading.current_thread())
        log.debug(salt_data['tag'])
        log.debug(salt_data)
        parts = salt_data['tag'].split('/')
        if len(parts) < 2:
            return
        if parts[1] == 'job':
            log.debug('In job part 1')
            if parts[3] == 'new':
                log.debug('In new job')
                self.process_new_job_event(salt_data)
            elif parts[3] == 'ret':
                log.debug('In ret')
                self.process_ret_job_event(salt_data)
                if salt_data['data']['fun'] == 'grains.items':
                    self.process_minion_update(salt_data)
        elif parts[1] == 'key':
            log.debug('In key')
            self.process_key_event(salt_data)
        elif parts[1] == 'presence':
            self.process_presence_events(salt_data, token, opts)