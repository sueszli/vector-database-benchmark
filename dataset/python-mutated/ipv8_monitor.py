import logging
import os
import time
from asyncio import ensure_future
from tribler_apptester.utils.asyncio import looping_call

class IPv8Monitor(object):
    """
    This class is responsible for monitoring IPv8 within Tribler.
    Specifically, it fetches information from the Tribler core and writes it to a file.
    """

    def __init__(self, request_manager, interval):
        if False:
            i = 10
            return i + 15
        self._logger = logging.getLogger(self.__class__.__name__)
        self.interval = interval
        self.request_manager = request_manager
        self.monitor_lc = None
        self.start_time = time.time()
        output_dir = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.overlay_stats_file_path = os.path.join(output_dir, 'ipv8_overlay_stats.csv')
        with open(self.overlay_stats_file_path, 'w') as output_file:
            output_file.write('time,overlay_id,num_peers\n')

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start the monitoring loop for the downloads.\n        '
        self._logger.info('Starting IPv8 monitor (interval: %d seconds)' % self.interval)
        self.monitor_lc = ensure_future(looping_call(0, self.interval, self.monitor_ipv8))

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop the monitoring loop for the downloads.\n        '
        if self.monitor_lc:
            self.monitor_lc.cancel()
            self.monitor_lc = None

    async def monitor_ipv8(self):
        """
        Monitor IPv8.
        """
        statistics = await self.request_manager.get_overlay_statistics()
        if 'overlays' not in statistics:
            return
        for overlay in statistics['overlays']:
            with open(self.overlay_stats_file_path, 'a') as output_file:
                time_diff = time.time() - self.start_time
                output_file.write('%s,%s,%d\n' % (time_diff, overlay['master_peer'][-6:], len(overlay['peers'])))