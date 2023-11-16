from __future__ import absolute_import, unicode_literals
from itertools import cycle
import math
from pokemongo_bot.walkers.step_walker import StepWalker
from pokemongo_bot.base_task import BaseTask

class FollowSpiral(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1

    def initialize(self):
        if False:
            return 10
        self.steplimit = self.config.get('diameter', 4)
        self.step_size = self.config.get('step_size', 70)
        self.origin_lat = self.bot.position[0]
        self.origin_lon = self.bot.position[1]
        self.diameter_to_steps = (self.steplimit + 1) ** 2
        self.spiral = self._generate_spiral(self.origin_lat, self.origin_lon, self.step_size, self.diameter_to_steps)
        self.points = cycle(self.spiral + list(reversed(self.spiral))[1:-1])
        self.next_point = None

    @staticmethod
    def _generate_spiral(starting_lat, starting_lng, step_size, step_limit):
        if False:
            i = 10
            return i + 15
        '\n        Sourced from:\n        https://github.com/tejado/pgoapi/blob/master/examples/spiral_poi_search.py\n\n        :param starting_lat:\n        :param starting_lng:\n        :param step_size:\n        :param step_limit:\n        :return:\n        '
        coords = [{'lat': starting_lat, 'lng': starting_lng}]
        (steps, x, y, d, m) = (1, 0, 0, 1, 1)
        rlat = starting_lat * math.pi
        latdeg = 111132.93 - 559.82 * math.cos(2 * rlat) + 1.175 * math.cos(4 * rlat)
        lngdeg = 111412.84 * math.cos(rlat) - 93.5 * math.cos(3 * rlat)
        step_size_lat = step_size / latdeg
        step_size_lng = step_size / lngdeg
        while steps < step_limit:
            while 2 * x * d < m and steps < step_limit:
                x = x + d
                steps += 1
                lat = x * step_size_lat + starting_lat
                lng = y * step_size_lng + starting_lng
                coords.append({'lat': lat, 'lng': lng})
            while 2 * y * d < m and steps < step_limit:
                y = y + d
                steps += 1
                lat = x * step_size_lat + starting_lat
                lng = y * step_size_lng + starting_lng
                coords.append({'lat': lat, 'lng': lng})
            d *= -1
            m += 1
        return coords

    def work(self):
        if False:
            while True:
                i = 10
        if not self.next_point:
            self.next_point = next(self.points)
        point = self.next_point
        step_walker = StepWalker(self.bot, point['lat'], point['lng'])
        if step_walker.step():
            self.next_point = None
        return [point['lat'], point['lng']]