import sampy
from gavi import logging as logging_
import threading
logger = logging_.getLogger('gavi.samp')

class Samp(object):

    def __init__(self, daemon=True, name=None):
        if False:
            i = 10
            return i + 15
        self.client = sampy.SAMPIntegratedClient(metadata={'samp.name': 'Gavi client' if name is None else name, 'samp.description.text': 'Gavi client' if name is None else name, 'gavi.samp.version': '0.01'}, callable=True)

        def _myrun_client():
            if False:
                for i in range(10):
                    print('nop')
            if self.client.client._callable:
                self.client.client._thread = threading.Thread(target=self.client.client._serve_forever)
                self.client.client._thread.setDaemon(True)
                self.client.client._thread.start()
        if daemon:
            self.client.client._run_client = _myrun_client
        connected = False
        try:
            self.client.connect()
            connected = True
        except sampy.SAMPHubError as e:
            print(('error connecting to hub', e))
        if connected:
            logger.info('connected to SAMP hub')
            logger.info('binding events')
            self.client.bindReceiveCall('table.load.votable', self._onTableLoadVotable)
            self.client.bindReceiveNotification('table.load.votable', self._onTableLoadVotable)
        self.tableLoadCallbacks = []

    def _onTableLoadVotable(self, private_key, sender_id, msg_id, mtype, params, extra):
        if False:
            i = 10
            return i + 15
        print(('Msg:', repr(private_key), repr(sender_id), repr(msg_id), repr(mtype), repr(params), repr(extra)))
        try:
            url = params['url']
            table_id = params['table-id']
            name = params['name']
            for callback in self.tableLoadCallbacks:
                callback(url, table_id, name)
        except:
            logger.exception('event handler failed')
        if msg_id != None:
            self.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result={'txt': 'loaded'})

    def _onSampNotification(self, private_key, sender_id, mtype, params, extra):
        if False:
            print('Hello World!')
        print(('Notification:', repr(private_key), repr(sender_id), repr(mtype), repr(params), repr(extra)))

    def _onSampCall(self, private_key, sender_id, msg_id, mtype, params, extra):
        if False:
            return 10
        print('----')
        try:
            print(('Call:', repr(private_key), repr(sender_id), repr(msg_id), repr(mtype), repr(params), repr(extra)))
            self.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result={'txt': 'printed'})
        except:
            print('errrrrrrororrrr hans!')