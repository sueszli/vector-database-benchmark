"""HDD temperature plugin."""
import os
import socket
from glances.globals import nativestr
from glances.logger import logger
from glances.plugins.plugin.model import GlancesPluginModel

class PluginModel(GlancesPluginModel):
    """Glances HDD temperature sensors plugin.

    stats is a list
    """

    def __init__(self, args=None, config=None):
        if False:
            return 10
        'Init the plugin.'
        super(PluginModel, self).__init__(args=args, config=config, stats_init_value=[])
        hddtemp_host = self.get_conf_value('host', default=['127.0.0.1'])[0]
        hddtemp_port = int(self.get_conf_value('port', default='7634'))
        self.hddtemp = GlancesGrabHDDTemp(args=args, host=hddtemp_host, port=hddtemp_port)
        self.display_curse = False

    @GlancesPluginModel._log_result_decorator
    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'Update HDD stats using the input method.'
        stats = self.get_init_value()
        if self.input_method == 'local':
            stats = self.hddtemp.get()
        else:
            pass
        self.stats = stats
        return self.stats

class GlancesGrabHDDTemp(object):
    """Get hddtemp stats using a socket connection."""

    def __init__(self, host='127.0.0.1', port=7634, args=None):
        if False:
            while True:
                i = 10
        'Init hddtemp stats.'
        self.args = args
        self.host = host
        self.port = port
        self.cache = ''
        self.reset()

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Reset/init the stats.'
        self.hddtemp_list = []

    def __update__(self):
        if False:
            return 10
        'Update the stats.'
        self.reset()
        data = self.fetch()
        if data == '':
            return
        if len(data) < 14:
            data = self.cache if len(self.cache) > 0 else self.fetch()
        self.cache = data
        try:
            fields = data.split(b'|')
        except TypeError:
            fields = ''
        devices = (len(fields) - 1) // 5
        for item in range(devices):
            offset = item * 5
            hddtemp_current = {}
            device = os.path.basename(nativestr(fields[offset + 1]))
            temperature = fields[offset + 3]
            unit = nativestr(fields[offset + 4])
            hddtemp_current['label'] = device
            try:
                hddtemp_current['value'] = float(temperature)
            except ValueError:
                hddtemp_current['value'] = nativestr(temperature)
            hddtemp_current['unit'] = unit
            self.hddtemp_list.append(hddtemp_current)

    def fetch(self):
        if False:
            while True:
                i = 10
        'Fetch the data from hddtemp daemon.'
        try:
            sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sck.connect((self.host, self.port))
            data = b''
            while True:
                received = sck.recv(4096)
                if not received:
                    break
                data += received
        except Exception as e:
            logger.debug('Cannot connect to an HDDtemp server ({}:{} => {})'.format(self.host, self.port, e))
            logger.debug('Disable the HDDtemp module. Use the --disable-hddtemp to hide the previous message.')
            if self.args is not None:
                self.args.disable_hddtemp = True
            data = ''
        finally:
            sck.close()
            if data != '':
                logger.debug('Received data from the HDDtemp server: {}'.format(data))
        return data

    def get(self):
        if False:
            return 10
        'Get HDDs list.'
        self.__update__()
        return self.hddtemp_list