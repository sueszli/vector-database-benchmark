"""
 @file
 @brief Process project data, scaling keyframe X coordinates by the given factor
 @author Jonathan Thomas <jonathan@openshot.org>
 @author FeRD (Frank Dana) <ferdnyc@gmail.com>

 @section LICENSE

 Copyright (c) 2008-2020 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """
from classes.logger import log

class KeyframeScaler:
    """This factory class produces scaler objects which, when called,
    will apply the assigned scaling factor to the keyframe points
    in a project data dictionary. Keyframe X coordinate values are
    multiplied by the scaling factor, except X=1 (because the first
    frame never changes)"""

    def _scale_value(self, value: float) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Scale value by some factor, except for 1 (leave that alone)'
        if value == 1.0:
            return value
        return round(value * self._scale_factor)

    def _update_prop(self, prop: dict, scale_y=False):
        if False:
            i = 10
            return i + 15
        'To keep keyframes at the same time in video,\n        update frame numbers to the new framerate.\n\n        scale_y: if the y coordinate also represents a frame number,\n        this flag will scale both x and y.\n        '
        if 'red' in prop:
            keyframes = [prop[color].get('Points', []) for color in prop]
        else:
            keyframes = [prop.get('Points', [])]
        for k in keyframes:
            if scale_y:
                log.debug('Updating x and y coordinates of time keyframes')
                [point['co'].update({'X': self._scale_value(point['co'].get('X', 0.0)), 'Y': self._scale_value(point['co'].get('Y', 0.0))}) for point in k if 'co' in point]
            else:
                [point['co'].update({'X': self._scale_value(point['co'].get('X', 0.0))}) for point in k if 'co' in point]

    def _process_item(self, item: dict):
        if False:
            print('Hello World!')
        'Process all the dict sub-members of the current dict'
        props = [prop for prop in item if isinstance(item[prop], dict)]
        for prop_name in props:
            self._update_prop(item[prop_name], scale_y=prop_name == 'time')

    def __call__(self, data: dict) -> dict:
        if False:
            return 10
        'Apply the stored scaling factor to a project data dict'
        for clip in data.get('clips', []):
            self._process_item(clip)
            for effect in clip.get('effects', []):
                self._process_item(effect)
        for effect in data.get('effects', []):
            self._process_item(effect)
        return data

    def __init__(self, factor: float):
        if False:
            for i in range(10):
                print('nop')
        'Store the scale factor assigned to this instance'
        self._scale_factor = factor