import random
import signal
from itertools import cycle
from typing import List, NamedTuple
import envoy
from mode import Service, Worker

class Span(NamedTuple):
    length: int
    signals: List[int]
    min_sleep: float
    max_sleep: float

    def seconds_to_sleep(self):
        if False:
            while True:
                i = 10
        return random.uniform(self.min_sleep, self.max_sleep)

class Chaos(Service):
    schedule = [Span(100, signals=[signal.SIGTERM, signal.SIGINT], min_sleep=1.0, max_sleep=30.0), Span(50, signals=[signal.SIGTERM], min_sleep=5.0, max_sleep=100.0), Span(30, signals=[signal.SIGKILL], min_sleep=0.1, max_sleep=1.0)]

    def iterate_over_scheduled_time(self):
        if False:
            return 10
        for period in cycle(self.schedule):
            for _ in range(period.length):
                yield period

    @Service.task
    async def _runs_in_background(self):
        self.log.info("I'm your local friendly chaos monkey tester")
        self.log.info('Starting signal dispatcher.')
        for current_span in self.iterate_over_scheduled_time():
            if self.should_stop:
                return
            secs = current_span.seconds_to_sleep()
            self.log.info('Signal dispatcher sleeping for %r seconds...', secs)
            await self.sleep(secs)
            sig = random.choice(current_span.signals)
            self.log.warning('Signalling all workers on this box with %r', sig)
            r = envoy.run(f'pkill -{int(sig)} Faust:Worker')
            if r.status_code:
                if r.std_err.strip():
                    self.log.error('ERROR from pkill: %r', r.std_err)
                else:
                    self.log.info('No processes running, nothing to signal!')
if __name__ == '__main__':
    Worker(Chaos(), loglevel='INFO').execute_from_commandline()