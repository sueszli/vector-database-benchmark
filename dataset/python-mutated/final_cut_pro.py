"""
 @file
 @brief This file is used to import a Final Cut Pro XML file
 @author Jonathan Thomas <jonathan@openshot.org>

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
import json
import os
from operator import itemgetter
from xml.dom import minidom
import openshot
from PyQt5.QtWidgets import QFileDialog
from classes import info
from classes.app import get_app
from classes.logger import log
from classes.image_types import get_media_type
from classes.query import Clip, Track, File
from windows.views.find_file import find_missing_file

def import_xml():
    if False:
        for i in range(10):
            print('nop')
    'Import final cut pro XML file'
    app = get_app()
    _ = app._tr
    fps_num = app.project.get('fps').get('num', 24)
    fps_den = app.project.get('fps').get('den', 1)
    fps_float = float(fps_num / fps_den)
    recommended_path = app.project.current_filepath or ''
    if not recommended_path:
        recommended_path = info.HOME_PATH
    else:
        recommended_path = os.path.dirname(recommended_path)
    file_path = QFileDialog.getOpenFileName(app.window, _('Import XML...'), recommended_path, _('Final Cut Pro (*.xml)'), _('Final Cut Pro (*.xml)'))[0]
    if not file_path or not os.path.exists(file_path):
        return
    xmldoc = minidom.parse(file_path)
    video_tracks = []
    for video_element in xmldoc.getElementsByTagName('video'):
        for video_track in video_element.getElementsByTagName('track'):
            video_tracks.append(video_track)
    audio_tracks = []
    for audio_element in xmldoc.getElementsByTagName('audio'):
        for audio_track in audio_element.getElementsByTagName('track'):
            audio_tracks.append(audio_track)
    track_index = 0
    for tracks in [audio_tracks, video_tracks]:
        for track_element in tracks:
            clips_on_track = track_element.getElementsByTagName('clipitem')
            if not clips_on_track:
                continue
            track_index += 1
            all_tracks = app.project.get('layers')
            track_number = list(reversed(sorted(all_tracks, key=itemgetter('number'))))[0].get('number') + 1000000
            track = Track()
            is_locked = False
            if track_element.getElementsByTagName('locked')[0].childNodes[0].nodeValue == 'TRUE':
                is_locked = True
            track.data = {'number': track_number, 'y': 0, 'label': 'XML Import %s' % track_index, 'lock': is_locked}
            track.save()
            for clip_element in clips_on_track:
                xml_file_id = clip_element.getElementsByTagName('file')[0].getAttribute('id')
                clip_path = ''
                if clip_element.getElementsByTagName('pathurl'):
                    clip_path = clip_element.getElementsByTagName('pathurl')[0].childNodes[0].nodeValue
                else:
                    continue
                (clip_path, is_modified, is_skipped) = find_missing_file(clip_path)
                if is_skipped:
                    continue
                file = File.get(path=clip_path)
                clip_obj = openshot.Clip(clip_path)
                if not file:
                    try:
                        reader = clip_obj.Reader()
                        file_data = json.loads(reader.Json())
                        file_data['media_type'] = get_media_type(file_data)
                        file = File()
                        file.data = file_data
                        file.save()
                    except Exception:
                        log.warning('Error building File object for %s' % clip_path, exc_info=1)
                if file.data['media_type'] == 'video' or file.data['media_type'] == 'image':
                    thumb_path = os.path.join(info.THUMBNAIL_PATH, '%s.png' % file.data['id'])
                else:
                    thumb_path = os.path.join(info.PATH, 'images', 'AudioThumbnail.png')
                clip = Clip()
                clip.data = json.loads(clip_obj.Json())
                clip.data['file_id'] = file.id
                clip.data['title'] = clip_element.getElementsByTagName('name')[0].childNodes[0].nodeValue
                clip.data['layer'] = track.data.get('number', 1000000)
                clip.data['image'] = thumb_path
                clip.data['position'] = float(clip_element.getElementsByTagName('start')[0].childNodes[0].nodeValue) / fps_float
                clip.data['start'] = float(clip_element.getElementsByTagName('in')[0].childNodes[0].nodeValue) / fps_float
                clip.data['end'] = float(clip_element.getElementsByTagName('out')[0].childNodes[0].nodeValue) / fps_float
                for effect_element in clip_element.getElementsByTagName('effect'):
                    effectid = effect_element.getElementsByTagName('effectid')[0].childNodes[0].nodeValue
                    keyframes = effect_element.getElementsByTagName('keyframe')
                    if effectid == 'opacity':
                        clip.data['alpha'] = {'Points': []}
                        for keyframe_element in keyframes:
                            keyframe_time = float(keyframe_element.getElementsByTagName('when')[0].childNodes[0].nodeValue)
                            keyframe_value = float(keyframe_element.getElementsByTagName('value')[0].childNodes[0].nodeValue) / 100.0
                            clip.data['alpha']['Points'].append({'co': {'X': round(keyframe_time), 'Y': keyframe_value}, 'interpolation': 1})
                    elif effectid == 'audiolevels':
                        clip.data['volume'] = {'Points': []}
                        for keyframe_element in keyframes:
                            keyframe_time = float(keyframe_element.getElementsByTagName('when')[0].childNodes[0].nodeValue)
                            keyframe_value = float(keyframe_element.getElementsByTagName('value')[0].childNodes[0].nodeValue) / 100.0
                            clip.data['volume']['Points'].append({'co': {'X': round(keyframe_time), 'Y': keyframe_value}, 'interpolation': 1})
                clip.save()
            app.window.refreshFrameSignal.emit()
            app.window.propertyTableView.select_frame(app.window.preview_thread.player.Position())
    xmldoc.unlink()