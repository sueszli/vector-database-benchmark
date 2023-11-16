"""
 @file
 @brief This file is used to import an EDL (edit decision list) file
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
import re
from operator import itemgetter
import openshot
from PyQt5.QtWidgets import QFileDialog
from classes import info
from classes.app import get_app
from classes.logger import log
from classes.image_types import get_media_type
from classes.query import Clip, Track, File
from classes.time_parts import timecodeToSeconds
from windows.views.find_file import find_missing_file
title_regex = re.compile('TITLE:[ ]+(.*)')
clips_regex = re.compile('(\\d{3})[ ]+(.+?)[ ]+(.+?)[ ]+(.+?)[ ]+(.*)[ ]+(.*)[ ]+(.*)[ ]+(.*)')
clip_name_regex = re.compile('[*][ ]+FROM CLIP NAME:[ ]+(.*)')
opacity_regex = re.compile('[*][ ]+OPACITY LEVEL AT (.*) IS [+-]*(.*)%')
audio_level_regex = re.compile('[*][ ]+AUDIO LEVEL AT (.*) IS [+]*(.*)[ ]+DB.*')
fcm_regex = re.compile('FCM:[ ]+(.*)')

def create_clip(context, track):
    if False:
        while True:
            i = 10
    'Create a new clip based on this context dict'
    app = get_app()
    _ = app._tr
    fps_num = app.project.get('fps').get('num', 24)
    fps_den = app.project.get('fps').get('den', 1)
    fps_float = float(fps_num / fps_den)
    (clip_path, is_modified, is_skipped) = find_missing_file(context.get('clip_path', ''))
    if is_skipped:
        return
    video_ctx = context.get('AX', {}).get('V', {})
    audio_ctx = context.get('AX', {}).get('A', {})
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
        except:
            log.warning('Error building File object for %s' % clip_path, exc_info=1)
    if file.data['media_type'] == 'video' or file.data['media_type'] == 'image':
        thumb_path = os.path.join(info.THUMBNAIL_PATH, '%s.png' % file.data['id'])
    else:
        thumb_path = os.path.join(info.PATH, 'images', 'AudioThumbnail.png')
    clip = Clip()
    clip.data = json.loads(clip_obj.Json())
    clip.data['file_id'] = file.id
    clip.data['title'] = context.get('clip_path', '')
    clip.data['layer'] = track.data.get('number', 1000000)
    if video_ctx and (not audio_ctx):
        clip.data['position'] = timecodeToSeconds(video_ctx.get('timeline_position', '00:00:00:00'), fps_num, fps_den)
        clip.data['start'] = timecodeToSeconds(video_ctx.get('clip_start_time', '00:00:00:00'), fps_num, fps_den)
        clip.data['end'] = timecodeToSeconds(video_ctx.get('clip_end_time', '00:00:00:00'), fps_num, fps_den)
        clip.data['has_audio'] = {'Points': [{'co': {'X': 1.0, 'Y': 0.0}, 'interpolation': 2}]}
    elif audio_ctx and (not video_ctx):
        clip.data['position'] = timecodeToSeconds(audio_ctx.get('timeline_position', '00:00:00:00'), fps_num, fps_den)
        clip.data['start'] = timecodeToSeconds(audio_ctx.get('clip_start_time', '00:00:00:00'), fps_num, fps_den)
        clip.data['end'] = timecodeToSeconds(audio_ctx.get('clip_end_time', '00:00:00:00'), fps_num, fps_den)
        clip.data['has_video'] = {'Points': [{'co': {'X': 1.0, 'Y': 0.0}, 'interpolation': 2}]}
    else:
        clip.data['position'] = timecodeToSeconds(video_ctx.get('timeline_position', '00:00:00:00'), fps_num, fps_den)
        clip.data['start'] = timecodeToSeconds(video_ctx.get('clip_start_time', '00:00:00:00'), fps_num, fps_den)
        clip.data['end'] = timecodeToSeconds(video_ctx.get('clip_end_time', '00:00:00:00'), fps_num, fps_den)
    if context.get('volume'):
        clip.data['volume'] = {'Points': []}
        for keyframe in context.get('volume', []):
            clip.data['volume']['Points'].append({'co': {'X': round(timecodeToSeconds(keyframe.get('time', 0.0), fps_num, fps_den) * fps_float), 'Y': keyframe.get('value', 0.0)}, 'interpolation': 1})
    if context.get('opacity'):
        clip.data['alpha'] = {'Points': []}
        for keyframe in context.get('opacity', []):
            clip.data['alpha']['Points'].append({'co': {'X': round(timecodeToSeconds(keyframe.get('time', 0.0), fps_num, fps_den) * fps_float), 'Y': keyframe.get('value', 0.0)}, 'interpolation': 1})
    clip.save()

def import_edl():
    if False:
        while True:
            i = 10
    'Import EDL File'
    app = get_app()
    _ = app._tr
    recommended_path = app.project.current_filepath or ''
    if not recommended_path:
        recommended_path = info.HOME_PATH
    else:
        recommended_path = os.path.dirname(recommended_path)
    file_path = QFileDialog.getOpenFileName(app.window, _('Import EDL...'), recommended_path, _('Edit Decision List (*.edl)'), _('Edit Decision List (*.edl)'))[0]
    if os.path.exists(file_path):
        context = {}
        current_clip_index = ''
        all_tracks = app.project.get('layers')
        track_number = list(reversed(sorted(all_tracks, key=itemgetter('number'))))[0].get('number') + 1000000
        track = Track()
        track.data = {'number': track_number, 'y': 0, 'label': 'EDL Import', 'lock': False}
        track.save()
        with open(file_path, 'r') as f:
            for line in f:
                for r in title_regex.findall(line):
                    context['title'] = r
                for r in clips_regex.findall(line):
                    if len(r) == 8:
                        edit_index = r[0]
                        tape = r[1]
                        clip_type = r[2]
                        if tape == 'BL':
                            continue
                        if current_clip_index == '':
                            current_clip_index = edit_index
                        if current_clip_index != edit_index:
                            create_clip(context, track)
                            current_clip_index = edit_index
                            context = {'title': context.get('title'), 'fcm': context.get('fcm')}
                        if tape not in context:
                            context[tape] = {}
                        if clip_type not in context[tape]:
                            context[tape][clip_type] = {}
                        context['edit_index'] = edit_index
                        context[tape][clip_type]['edit_type'] = r[3]
                        context[tape][clip_type]['clip_start_time'] = r[4]
                        context[tape][clip_type]['clip_end_time'] = r[5]
                        context[tape][clip_type]['timeline_position'] = r[6]
                        context[tape][clip_type]['timeline_position_end'] = r[7]
                for r in clip_name_regex.findall(line):
                    context['clip_path'] = r
                for r in opacity_regex.findall(line):
                    if len(r) == 2:
                        if 'opacity' not in context:
                            context['opacity'] = []
                        keyframe_time = r[0]
                        keyframe_value = float(r[1]) / 100.0
                        context['opacity'].append({'time': keyframe_time, 'value': keyframe_value})
                for r in audio_level_regex.findall(line):
                    if len(r) == 2:
                        if 'volume' not in context:
                            context['volume'] = []
                        keyframe_time = r[0]
                        keyframe_value = (float(r[1]) + 99.0) / 99.0
                        context['volume'].append({'time': keyframe_time, 'value': keyframe_value})
                for r in fcm_regex.findall(line):
                    context['fcm'] = r
            create_clip(context, track)
            app.window.refreshFrameSignal.emit()
            app.window.propertyTableView.select_frame(app.window.preview_thread.player.Position())