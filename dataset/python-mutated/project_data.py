"""
 @file
 @brief This file listens to changes, and updates the primary project data
 @author Noah Figg <eggmunkee@hotmail.com>
 @author Jonathan Thomas <jonathan@openshot.org>
 @author Olivier Girard <eolinwen@gmail.com>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
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
import copy
import glob
import os
import random
import shutil
import json
from classes import info
from classes.app import get_app
from classes.image_types import get_media_type
from classes.json_data import JsonDataStore
from classes.logger import log
from classes.updates import UpdateInterface
from classes.assets import get_assets_path
from windows.views.find_file import find_missing_file
from .keyframe_scaler import KeyframeScaler
import openshot

class ProjectDataStore(JsonDataStore, UpdateInterface):
    """ This class allows advanced searching of data structure, implements changes interface """

    def __init__(self):
        if False:
            while True:
                i = 10
        JsonDataStore.__init__(self)
        self.data_type = 'project data'
        self.default_project_filepath = os.path.join(info.PATH, 'settings', '_default.project')
        self.current_filepath = None
        self.has_unsaved_changes = False
        self.new()

    def needs_save(self):
        if False:
            while True:
                i = 10
        'Returns if project data has unsaved changes'
        return self.has_unsaved_changes

    def get(self, key):
        if False:
            return 10
        'Get copied value of a given key in data store'
        if not key:
            log.warning('ProjectDataStore cannot get empty key.')
            return None
        if not isinstance(key, list):
            key = [key]
        obj = self._data
        for key_index in range(len(key)):
            key_part = key[key_index]
            if not isinstance(key_part, dict) and (not isinstance(key_part, str)):
                log.error('Unexpected key part type: {}'.format(type(key_part).__name__))
                return None
            if isinstance(key_part, dict) and isinstance(obj, list):
                found = False
                for item_index in range(len(obj)):
                    item = obj[item_index]
                    match = True
                    for subkey in key_part:
                        subkey = subkey.lower()
                        if not (subkey in item and item[subkey] == key_part[subkey]):
                            match = False
                            break
                    if match:
                        found = True
                        obj = item
                        break
                if not found:
                    return None
            if isinstance(key_part, str):
                key_part = key_part.lower()
                if not isinstance(obj, dict):
                    log.warn('Invalid project data structure. Trying to use a key on a non-dictionary object. Key part: {} ("{}").\nKey: {}'.format(key_index, key_part, key))
                    return None
                if key_part not in obj:
                    log.warn('Key not found in project. Mismatch on key part %s ("%s").\nKey: %s', key_index, key_part, key)
                    return None
                obj = obj[key_part]
        return obj

    def set(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Prevent calling JsonDataStore set() method. It is not allowed in ProjectDataStore, as changes come from UpdateManager.'
        raise RuntimeError('ProjectDataStore.set() is not allowed. Changes must route through UpdateManager.')

    def _set(self, key, values=None, add=False, remove=False):
        if False:
            for i in range(10):
                print('nop')
        " Store setting, but adding isn't allowed. All possible settings must be in default settings file. "
        log.debug('_set key: %s, values: %s, add: %s, remove: %s', key, values, add, remove)
        (parent, my_key) = (None, '')
        if not isinstance(key, list):
            log.warning('_set() key must be a list. key=%s', key)
            return None
        if not key:
            log.warning('Cannot set empty key (key=%s)', key)
            return None
        obj = self._data
        for key_index in range(len(key)):
            key_part = key[key_index]
            if not isinstance(key_part, dict) and (not isinstance(key_part, str)):
                log.error('Unexpected key part type: %s', type(key_part).__name__)
                return None
            if isinstance(key_part, dict) and isinstance(obj, list):
                found = False
                for item_index in range(len(obj)):
                    item = obj[item_index]
                    match = True
                    for subkey in key_part.keys():
                        subkey = subkey.lower()
                        if not (subkey in item and item[subkey] == key_part[subkey]):
                            match = False
                            break
                    if match:
                        found = True
                        obj = item
                        my_key = item_index
                        break
                if not found:
                    return None
            if isinstance(key_part, str):
                key_part = key_part.lower()
                if not isinstance(obj, dict):
                    return None
                if key_part not in obj:
                    log.warn('Key not found in project. Mismatch on key part %s ("%s").\nKey: %s', key_index, key_part, key)
                    return None
                obj = obj[key_part]
                my_key = key_part
            if key_index < len(key) - 1 or key_index == 0:
                parent = obj
        ret = json.loads(json.dumps(obj))
        if remove:
            del parent[my_key]
        elif add and isinstance(parent, list):
            parent.append(values)
        elif isinstance(values, dict):
            obj.update(values)
        else:
            self._data[my_key] = values
        return ret

    def new(self):
        if False:
            return 10
        ' Try to load default project settings file, will raise error on failure '
        import openshot
        if os.path.exists(info.USER_DEFAULT_PROJECT):
            try:
                self._data = self.read_from_file(info.USER_DEFAULT_PROJECT)
            except (FileNotFoundError, PermissionError):
                log.warning('Unable to load user project defaults from %s', info.USER_DEFAULT_PROJECT, exc_info=1)
            except Exception:
                raise
            else:
                log.info('Loaded user project defaults from %s', info.USER_DEFAULT_PROJECT)
        else:
            self._data = self.read_from_file(self.default_project_filepath)
        self.current_filepath = None
        self.has_unsaved_changes = False
        info.reset_userdirs()
        s = get_app().get_settings()
        default_profile_desc = s.get('default-profile')
        profile = self.get_profile(profile_desc=default_profile_desc)
        if not profile:
            profile = self.get_profile(profile_desc='HD 720p 30 fps')
        if profile and default_profile_desc != profile.info.description:
            log.info(f'Updating default-profile from legacy `{default_profile_desc}` to `{profile.info.description}`.')
            s.set('default-profile', profile.info.description)
        self.apply_default_audio_settings()
        self._data['id'] = self.generate_id()

    def get_profile(self, profile_desc=None, profile_key=None):
        if False:
            for i in range(10):
                print('nop')
        'Attempt to find a specific profile'
        profile = None
        LEGACY_PROFILE_PATH = os.path.join(info.PROFILES_PATH, 'legacy')
        legacy_profile = None
        for legacy_filename in os.listdir(LEGACY_PROFILE_PATH):
            legacy_profile_path = os.path.join(LEGACY_PROFILE_PATH, legacy_filename)
            try:
                temp_profile = openshot.Profile(legacy_profile_path)
                if profile_desc == temp_profile.info.description:
                    legacy_profile = temp_profile
                    break
            except RuntimeError:
                pass
        profile_dirs = [info.USER_PROFILES_PATH, info.PROFILES_PATH]
        available_dirs = [f for f in profile_dirs if os.path.exists(f)]
        for profile_folder in available_dirs:
            for file in reversed(sorted(os.listdir(profile_folder))):
                profile_path = os.path.join(profile_folder, file)
                if os.path.isdir(profile_path):
                    continue
                try:
                    temp_profile = openshot.Profile(profile_path)
                    if profile_desc == temp_profile.info.description:
                        profile = self.apply_profile(temp_profile)
                        break
                    if legacy_profile and legacy_profile.Key() == temp_profile.Key():
                        profile = self.apply_profile(temp_profile)
                        break
                except RuntimeError as e:
                    log.error("Failed to parse file '%s' as a profile: %s" % (profile_path, e))
        return profile

    def apply_profile(self, profile):
        if False:
            print('Hello World!')
        'Apply a specific profile to the current project data'
        log.info('Setting profile to %s' % profile.info.description)
        self._data['profile'] = profile.info.description
        self._data['width'] = profile.info.width
        self._data['height'] = profile.info.height
        self._data['fps'] = {'num': profile.info.fps.num, 'den': profile.info.fps.den}
        self._data['display_ratio'] = {'num': profile.info.display_ratio.num, 'den': profile.info.display_ratio.den}
        self._data['pixel_ratio'] = {'num': profile.info.pixel_ratio.num, 'den': profile.info.pixel_ratio.den}
        return profile

    def load(self, file_path, clear_thumbnails=True):
        if False:
            for i in range(10):
                print('nop')
        ' Load project from file '
        self.new()
        if file_path:
            log.info('Loading project file: %s', file_path)
            default_project = self._data
            try:
                project_data = self.read_from_file(file_path, path_mode='absolute')
                if not project_data.get('history'):
                    project_data['history'] = {'undo': [], 'redo': []}
                get_app().window.actionClearWaveformData.setEnabled(False)
                for file in project_data['files']:
                    if file.get('ui', {}).get('audio_data', []):
                        get_app().window.actionClearWaveformData.setEnabled(True)
                        break
            except Exception:
                try:
                    project_data = self.read_legacy_project_file(file_path)
                except Exception:
                    raise
            self._data = self.merge_settings(default_project, project_data)
            self.current_filepath = file_path
            if clear_thumbnails:
                info.THUMBNAIL_PATH = os.path.join(get_assets_path(self.current_filepath), 'thumbnail')
                info.TITLE_PATH = os.path.join(get_assets_path(self.current_filepath), 'title')
                info.BLENDER_PATH = os.path.join(get_assets_path(self.current_filepath), 'blender')
                info.PROTOBUF_DATA_PATH = os.path.join(get_assets_path(self.current_filepath), 'protobuf_data')
            self.has_unsaved_changes = False
            self.check_if_paths_are_valid()
            openshot_thumbnails = info.get_default_path('THUMBNAIL_PATH')
            if os.path.exists(openshot_thumbnails) and clear_thumbnails:
                shutil.rmtree(openshot_thumbnails, True)
                os.mkdir(openshot_thumbnails)
            self.add_to_recent_files(file_path)
            self.upgrade_project_data_structures()
            project_profile_desc = self._data.get('profile', 'HD 720p 30 fps')
            profile = self.get_profile(profile_desc=project_profile_desc)
            if not profile:
                profile = self.get_profile(profile_desc='HD 720p 30 fps')
            self.apply_default_audio_settings()
        get_app().updates.load(self._data)

    def rescale_keyframes(self, scale_factor):
        if False:
            print('Hello World!')
        'Adjust all keyframe coordinates from previous FPS to new FPS (using a scale factor)\n           and return scaled project data without modifing the current project.'
        log.info('Scale all keyframes by a factor of %s', scale_factor)
        scaler = KeyframeScaler(factor=scale_factor)
        scaled = scaler(json.loads(json.dumps(self._data)))
        return scaled

    def read_legacy_project_file(self, file_path):
        if False:
            i = 10
            return i + 15
        'Attempt to read a legacy version 1.x openshot project file'
        import sys
        import pickle
        from classes.query import File, Track, Clip, Transition
        import openshot
        import json
        _ = get_app()._tr
        project_data = {}
        project_data['version'] = {'openshot-qt': info.VERSION, 'libopenshot': openshot.OPENSHOT_VERSION_FULL}
        fps = get_app().project.get('fps')
        fps_float = float(fps['num']) / float(fps['den'])
        from classes.legacy.openshot import classes as legacy_classes
        from classes.legacy.openshot.classes import project as legacy_project
        from classes.legacy.openshot.classes import sequences as legacy_sequences
        from classes.legacy.openshot.classes import track as legacy_track
        from classes.legacy.openshot.classes import clip as legacy_clip
        from classes.legacy.openshot.classes import keyframe as legacy_keyframe
        from classes.legacy.openshot.classes import files as legacy_files
        from classes.legacy.openshot.classes import transition as legacy_transition
        from classes.legacy.openshot.classes import effect as legacy_effect
        from classes.legacy.openshot.classes import marker as legacy_marker
        sys.modules['openshot.classes'] = legacy_classes
        sys.modules['classes.project'] = legacy_project
        sys.modules['classes.sequences'] = legacy_sequences
        sys.modules['classes.track'] = legacy_track
        sys.modules['classes.clip'] = legacy_clip
        sys.modules['classes.keyframe'] = legacy_keyframe
        sys.modules['classes.files'] = legacy_files
        sys.modules['classes.transition'] = legacy_transition
        sys.modules['classes.effect'] = legacy_effect
        sys.modules['classes.marker'] = legacy_marker
        failed_files = []
        with open(os.fsencode(file_path), 'rb') as f:
            try:
                v1_data = pickle.load(f, fix_imports=True, encoding='UTF-8')
                file_lookup = {}
                for item in v1_data.project_folder.items:
                    if isinstance(item, legacy_files.OpenShotFile):
                        try:
                            clip = openshot.Clip(item.name)
                            reader = clip.Reader()
                            file_data = json.loads(reader.Json(), strict=False)
                            file_data['media_type'] = get_media_type(file_data)
                            file = File()
                            file.data = file_data
                            file.save()
                            file_lookup[item.unique_id] = file
                        except Exception:
                            log.error('%s is not a valid video, audio, or image file', item.name, exc_info=1)
                            failed_files.append(item.name)
                track_list = Track.filter()
                for track in track_list:
                    track.delete()
                track_counter = 0
                for legacy_t in reversed(v1_data.sequences[0].tracks):
                    t = Track()
                    t.data = {'number': track_counter, 'y': 0, 'label': legacy_t.name}
                    t.save()
                    track_counter += 1
                track_counter = 0
                for sequence in v1_data.sequences:
                    for track in reversed(sequence.tracks):
                        for clip in track.clips:
                            if clip.file_object.unique_id in file_lookup:
                                file = file_lookup[clip.file_object.unique_id]
                            else:
                                log.info('Skipping importing missing file: %s' % clip.file_object.unique_id)
                                continue
                            if file.data['media_type'] == 'video' or file.data['media_type'] == 'image':
                                thumb_path = os.path.join(info.THUMBNAIL_PATH, '%s.png' % file.data['id'])
                            else:
                                thumb_path = os.path.join(info.PATH, 'images', 'AudioThumbnail.png')
                            filename = os.path.basename(file.data['path'])
                            file_path = file.absolute_path()
                            c = openshot.Clip(file_path)
                            new_clip = json.loads(c.Json(), strict=False)
                            new_clip['file_id'] = file.id
                            new_clip['title'] = filename
                            new_clip['start'] = clip.start_time
                            new_clip['end'] = clip.end_time
                            new_clip['position'] = clip.position_on_track
                            new_clip['layer'] = track_counter
                            if clip.video_fade_in or clip.video_fade_out:
                                new_clip['alpha']['Points'] = []
                            if clip.video_fade_in:
                                start = openshot.Point(round(clip.start_time * fps_float) + 1, 0.0, openshot.BEZIER)
                                start_object = json.loads(start.Json(), strict=False)
                                end = openshot.Point(round((clip.start_time + clip.video_fade_in_amount) * fps_float) + 1, 1.0, openshot.BEZIER)
                                end_object = json.loads(end.Json(), strict=False)
                                new_clip['alpha']['Points'].append(start_object)
                                new_clip['alpha']['Points'].append(end_object)
                            if clip.video_fade_out:
                                start = openshot.Point(round((clip.end_time - clip.video_fade_out_amount) * fps_float) + 1, 1.0, openshot.BEZIER)
                                start_object = json.loads(start.Json(), strict=False)
                                end = openshot.Point(round(clip.end_time * fps_float) + 1, 0.0, openshot.BEZIER)
                                end_object = json.loads(end.Json(), strict=False)
                                new_clip['alpha']['Points'].append(start_object)
                                new_clip['alpha']['Points'].append(end_object)
                            if clip.audio_fade_in or clip.audio_fade_out:
                                new_clip['volume']['Points'] = []
                            else:
                                p = openshot.Point(1, clip.volume / 100.0, openshot.BEZIER)
                                p_object = json.loads(p.Json(), strict=False)
                                new_clip['volume'] = {'Points': [p_object]}
                            if clip.audio_fade_in:
                                start = openshot.Point(round(clip.start_time * fps_float) + 1, 0.0, openshot.BEZIER)
                                start_object = json.loads(start.Json(), strict=False)
                                end = openshot.Point(round((clip.start_time + clip.video_fade_in_amount) * fps_float) + 1, clip.volume / 100.0, openshot.BEZIER)
                                end_object = json.loads(end.Json(), strict=False)
                                new_clip['volume']['Points'].append(start_object)
                                new_clip['volume']['Points'].append(end_object)
                            if clip.audio_fade_out:
                                start = openshot.Point(round((clip.end_time - clip.video_fade_out_amount) * fps_float) + 1, clip.volume / 100.0, openshot.BEZIER)
                                start_object = json.loads(start.Json(), strict=False)
                                end = openshot.Point(round(clip.end_time * fps_float) + 1, 0.0, openshot.BEZIER)
                                end_object = json.loads(end.Json(), strict=False)
                                new_clip['volume']['Points'].append(start_object)
                                new_clip['volume']['Points'].append(end_object)
                            clip_object = Clip()
                            clip_object.data = new_clip
                            clip_object.save()
                        for trans in track.transitions:
                            if not trans.resource or not os.path.exists(trans.resource):
                                trans.resource = os.path.join(info.PATH, 'transitions', 'common', 'fade.svg')
                            transition_reader = openshot.QtImageReader(trans.resource)
                            trans_begin_value = 1.0
                            trans_end_value = -1.0
                            if trans.reverse:
                                trans_begin_value = -1.0
                                trans_end_value = 1.0
                            brightness = openshot.Keyframe()
                            brightness.AddPoint(1, trans_begin_value, openshot.BEZIER)
                            brightness.AddPoint(round(trans.length * fps_float) + 1, trans_end_value, openshot.BEZIER)
                            contrast = openshot.Keyframe(trans.softness * 10.0)
                            transitions_data = {'id': get_app().project.generate_id(), 'layer': track_counter, 'title': 'Transition', 'type': 'Mask', 'position': trans.position_on_track, 'start': 0, 'end': trans.length, 'brightness': json.loads(brightness.Json(), strict=False), 'contrast': json.loads(contrast.Json(), strict=False), 'reader': json.loads(transition_reader.Json(), strict=False), 'replace_image': False}
                            t = Transition()
                            t.data = transitions_data
                            t.save()
                        track_counter += 1
            except Exception as ex:
                msg = 'Failed to load legacy project file %(path)s' % {'path': file_path}
                log.error(msg, exc_info=1)
                raise RuntimeError(msg) from ex
        if failed_files:
            raise RuntimeError('Failed to load the following files:\n%s' % ', '.join(failed_files))
        log.info('Successfully loaded legacy project file: %s', file_path)
        return project_data

    def upgrade_project_data_structures(self):
        if False:
            i = 10
            return i + 15
        'Fix any issues with old project files (if any)'
        openshot_version = self._data['version']['openshot-qt']
        libopenshot_version = self._data['version']['libopenshot']
        log.info('Project data: openshot %s, libopenshot %s', openshot_version, libopenshot_version)
        if openshot_version == '0.0.0':
            for clip in self._data['clips']:
                for point in clip['alpha']['Points']:
                    if 'co' in point:
                        point['co']['Y'] = 1.0 - point['co']['Y']
                    if 'handle_left' in point:
                        point['handle_left']['Y'] = 1.0 - point['handle_left']['Y']
                    if 'handle_right' in point:
                        point['handle_right']['Y'] = 1.0 - point['handle_right']['Y']
        elif openshot_version <= '2.1.0-dev':
            for clip_type in ['clips', 'effects']:
                for clip in self._data[clip_type]:
                    for object in [clip] + clip.get('effects', []):
                        for (item_key, item_data) in object.items():
                            if type(item_data) == dict and 'Points' in item_data:
                                for point in item_data.get('Points'):
                                    if 'handle_left' in point:
                                        point.get('handle_left')['X'] = 0.5
                                        point.get('handle_left')['Y'] = 1.0
                                    if 'handle_right' in point:
                                        point.get('handle_right')['X'] = 0.5
                                        point.get('handle_right')['Y'] = 0.0
                            elif type(item_data) == dict and 'red' in item_data:
                                for color in ['red', 'blue', 'green', 'alpha']:
                                    for point in item_data.get(color).get('Points'):
                                        if 'handle_left' in point:
                                            point.get('handle_left')['X'] = 0.5
                                            point.get('handle_left')['Y'] = 1.0
                                        if 'handle_right' in point:
                                            point.get('handle_right')['X'] = 0.5
                                            point.get('handle_right')['Y'] = 0.0
        elif openshot_version.startswith('2.5.'):
            log.debug('Scanning OpenShot 2.5 project for legacy cropping')
            for clip in self._data.get('clips', []):
                crop_x = clip.pop('crop_x', {})
                crop_y = clip.pop('crop_y', {})
                crop_width = clip.pop('crop_width', {})
                crop_height = clip.pop('crop_height', {})
                if any([self.is_keyframe_valid(crop_x, 0.0), self.is_keyframe_valid(crop_y, 0.0), self.is_keyframe_valid(crop_width, 1.0), self.is_keyframe_valid(crop_height, 1.0)]):
                    log.info('Migrating OpenShot 2.5 crop properties for clip %s', clip.get('id', '<unknown>'))
                    from json import loads as jl
                    effect = openshot.EffectInfo().CreateEffect('Crop')
                    effect.Id(get_app().project.generate_id())
                    effect_json = jl(effect.Json())
                    effect_json.update({'x': crop_x or jl(openshot.Keyframe(0.0).Json()), 'y': crop_y or jl(openshot.Keyframe(0.0).Json()), 'right': crop_width or jl(openshot.Keyframe(1.0).Json()), 'bottom': crop_height or jl(openshot.Keyframe(1.0).Json())})
                    for prop in ['right', 'bottom']:
                        for point in effect_json[prop].get('Points', []):
                            point['co']['Y'] = 1.0 - point.get('co', {}).get('Y', 0.0)
                    clip['effects'].append(effect_json)
        if self._data.get('id') == 'T0':
            self._data['id'] = self.generate_id()

    def is_keyframe_valid(self, keyframe, default_value):
        if False:
            i = 10
            return i + 15
        'Check if a keyframe is not empty (i.e. > 1 point, or a non default_value)'
        points = keyframe.get('Points', [])
        if not points or not isinstance(points, list):
            return False
        return any([len(points) > 1, points[0].get('co', {}).get('Y', default_value) != default_value])

    def save(self, file_path, backup_only=False):
        if False:
            return 10
        ' Save project file to disk '
        import openshot
        log.info('Saving project file: %s', file_path)
        if not backup_only:
            self.move_temp_paths_to_project_folder(file_path, previous_path=self.current_filepath)
        self._data['version'] = {'openshot-qt': info.VERSION, 'libopenshot': openshot.OPENSHOT_VERSION_FULL}
        self.write_to_file(file_path, self._data, path_mode='ignore' if backup_only else 'relative', previous_path=self.current_filepath if not backup_only else None)
        if not backup_only:
            self.current_filepath = file_path
            info.THUMBNAIL_PATH = os.path.join(get_assets_path(self.current_filepath), 'thumbnail')
            info.TITLE_PATH = os.path.join(get_assets_path(self.current_filepath), 'title')
            info.BLENDER_PATH = os.path.join(get_assets_path(self.current_filepath), 'blender')
            self.add_to_recent_files(file_path)
            self.has_unsaved_changes = False

    def move_temp_paths_to_project_folder(self, file_path, previous_path=None):
        if False:
            while True:
                i = 10
        ' Move all temp files (such as Thumbnails, Titles, and Blender animations) to the project asset folder. '
        try:
            asset_path = get_assets_path(file_path)
            target_thumb_path = os.path.join(asset_path, 'thumbnail')
            target_title_path = os.path.join(asset_path, 'title')
            target_blender_path = os.path.join(asset_path, 'blender')
            target_protobuf_path = os.path.join(asset_path, 'protobuf_data')
            try:
                for target_dir in [asset_path, target_thumb_path, target_title_path, target_blender_path, target_protobuf_path]:
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)
            except OSError:
                pass
            if previous_path:
                previous_asset_path = get_assets_path(previous_path)
                info.THUMBNAIL_PATH = os.path.join(previous_asset_path, 'thumbnail')
                info.TITLE_PATH = os.path.join(previous_asset_path, 'title')
                info.BLENDER_PATH = os.path.join(previous_asset_path, 'blender')
                info.PROTOBUF_DATA_PATH = os.path.join(previous_asset_path, 'protobuf_data')
            copied = []
            reader_paths = {}
            for thumb_path in os.listdir(info.THUMBNAIL_PATH):
                working_thumb_path = os.path.join(info.THUMBNAIL_PATH, thumb_path)
                target_thumb_filepath = os.path.join(target_thumb_path, thumb_path)
                if not os.path.exists(target_thumb_filepath):
                    shutil.copy2(working_thumb_path, target_thumb_filepath)
            for title_path in os.listdir(info.TITLE_PATH):
                working_title_path = os.path.join(info.TITLE_PATH, title_path)
                target_title_filepath = os.path.join(target_title_path, title_path)
                if not os.path.exists(target_title_filepath):
                    shutil.copy2(working_title_path, target_title_filepath)
            for blender_path in os.listdir(info.BLENDER_PATH):
                working_blender_path = os.path.join(info.BLENDER_PATH, blender_path)
                target_blender_filepath = os.path.join(target_blender_path, blender_path)
                if os.path.isdir(working_blender_path) and (not os.path.exists(target_blender_filepath)):
                    shutil.copytree(working_blender_path, target_blender_filepath)
            for protobuf_path in os.listdir(info.PROTOBUF_DATA_PATH):
                working_protobuf_path = os.path.join(info.PROTOBUF_DATA_PATH, protobuf_path)
                target_protobuf_filepath = os.path.join(target_protobuf_path, protobuf_path)
                if not os.path.exists(target_protobuf_filepath):
                    shutil.copy2(working_protobuf_path, target_protobuf_filepath)
            for file in self._data['files']:
                path = file['path']
                file_id = file['id']
                file['image'] = os.path.join(target_thumb_path, f'{file_id}.png')
                new_asset_path = None
                if info.BLENDER_PATH in path:
                    log.info('Copying %s', path)
                    (old_dir, asset_name) = os.path.split(path)
                    if os.path.isdir(old_dir) and old_dir not in copied:
                        old_dir_name = os.path.basename(old_dir)
                        copied.append(old_dir)
                        log.info('Copied dir %s to %s', old_dir_name, target_blender_path)
                    new_asset_path = os.path.join(target_blender_path, old_dir_name, asset_name)
                if info.TITLE_PATH in path:
                    log.info('Copying %s', path)
                    (old_dir, asset_name) = os.path.split(path)
                    if asset_name not in copied:
                        copied.append(asset_name)
                        log.info('Copied title %s to %s', asset_name, target_title_path)
                    new_asset_path = os.path.join(target_title_path, asset_name)
                if new_asset_path:
                    file['path'] = new_asset_path
                    reader_paths[file_id] = new_asset_path
                    log.info('Set file %s path to %s', file_id, new_asset_path)
            for clip in self._data['clips']:
                file_id = clip['file_id']
                clip_id = clip['id']
                clip['image'] = os.path.join(target_thumb_path, f'{file_id}.png')
                log.info('Checking clip %s path for file %s', clip_id, file_id)
                if file_id and file_id in reader_paths:
                    clip['reader']['path'] = reader_paths[file_id]
                    log.info('Updated clip %s path for file %s', clip_id, file_id)
                log.info('Checking effects in clip %s path for protobuf files' % clip_id)
                for effect in clip.get('effects', []):
                    if 'protobuf_data_path' in effect:
                        old_protobuf_path = effect['protobuf_data_path']
                        (old_protobuf_dir, protobuf_name) = os.path.split(old_protobuf_path)
                        if old_protobuf_dir != target_protobuf_path:
                            effect['protobuf_data_path'] = os.path.join(target_protobuf_path, protobuf_name)
                            log.info('Copied protobuf %s to %s', old_protobuf_path, target_protobuf_path)
        except Exception:
            log.error('Error while moving temp paths to project assets folder %s', asset_path, exc_info=1)

    def add_to_recent_files(self, file_path):
        if False:
            return 10
        ' Add this project to the recent files list '
        if not file_path or file_path is info.BACKUP_FILE:
            return
        s = get_app().get_settings()
        recent_projects = s.get('recent_projects')
        file_path = os.path.abspath(file_path)
        if file_path in recent_projects:
            recent_projects.remove(file_path)
        if len(recent_projects) > 10:
            del recent_projects[0]
        recent_projects.append(file_path)
        s.set('recent_projects', recent_projects)
        s.save()

    def check_if_paths_are_valid(self):
        if False:
            i = 10
            return i + 15
        'Check if all paths are valid, and prompt to update them if needed'
        app = get_app()
        settings = app.get_settings()
        _ = app._tr
        log.info('checking project files...')
        for file in reversed(self._data['files']):
            path = file['path']
            (parent_path, file_name_with_ext) = os.path.split(path)
            log.info('checking file %s', path)
            if not os.path.exists(path) and '%' not in path:
                (path, is_modified, is_skipped) = find_missing_file(path)
                if path and is_modified and (not is_skipped):
                    file['path'] = path
                    settings.setDefaultPath(settings.actionType.IMPORT, path)
                    log.info('Auto-updated missing file: %s', path)
                elif is_skipped:
                    log.info('Removed missing file: %s', file_name_with_ext)
                    self._data['files'].remove(file)
        for clip in reversed(self._data['clips']):
            path = clip.get('reader', {}).get('path', '')
            if path and (not os.path.exists(path)) and ('%' not in path):
                (path, is_modified, is_skipped) = find_missing_file(path)
                file_name_with_ext = os.path.basename(path)
                if path and is_modified and (not is_skipped):
                    clip['reader']['path'] = path
                    log.info('Auto-updated missing file: %s', clip['reader']['path'])
                elif is_skipped:
                    log.info('Removed missing clip: %s', file_name_with_ext)
                    self._data['clips'].remove(clip)

    def changed(self, action):
        if False:
            print('Hello World!')
        ' This method is invoked by the UpdateManager each time a change happens (i.e UpdateInterface) '
        if action.type == 'insert':
            old_vals = self._set(action.key, action.values, add=True)
            action.set_old_values(old_vals)
            self.has_unsaved_changes = True
        elif action.type == 'update':
            old_vals = self._set(action.key, action.values)
            action.set_old_values(old_vals)
            self.has_unsaved_changes = True
        elif action.type == 'delete':
            old_vals = self._set(action.key, remove=True)
            action.set_old_values(old_vals)
            self.has_unsaved_changes = True
        elif action.type == 'load':
            pass

    def generate_id(self, digits=10):
        if False:
            return 10
        ' Generate random alphanumeric ids '
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        id = ''
        for i in range(digits):
            c_index = random.randint(0, len(chars) - 1)
            id += chars[c_index]
        return id

    def apply_default_audio_settings(self):
        if False:
            while True:
                i = 10
        'Apply the default preferences for sampleRate and channels to\n        the current project data, to force playback at a specific rate and for\n        a specific # of audio channels and channel layout.'
        s = get_app().get_settings()
        default_sample_rate = int(s.get('default-samplerate'))
        default_channel_layout = s.get('default-channellayout')
        channels = 2
        channel_layout = openshot.LAYOUT_STEREO
        if default_channel_layout == 'LAYOUT_MONO':
            channels = 1
            channel_layout = openshot.LAYOUT_MONO
        elif default_channel_layout == 'LAYOUT_STEREO':
            channels = 2
            channel_layout = openshot.LAYOUT_STEREO
        elif default_channel_layout == 'LAYOUT_SURROUND':
            channels = 3
            channel_layout = openshot.LAYOUT_SURROUND
        elif default_channel_layout == 'LAYOUT_5POINT1':
            channels = 6
            channel_layout = openshot.LAYOUT_5POINT1
        elif default_channel_layout == 'LAYOUT_7POINT1':
            channels = 8
            channel_layout = openshot.LAYOUT_7POINT1
        self._data['sample_rate'] = default_sample_rate
        self._data['channels'] = channels
        self._data['channel_layout'] = channel_layout
        log.info('Apply default audio playback settings: %s, %s channels' % (self._data['sample_rate'], self._data['channels']))