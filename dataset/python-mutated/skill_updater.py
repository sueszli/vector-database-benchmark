"""Periodically run by skill manager to update skills and post the manifest."""
import os
import sys
from datetime import datetime
from time import time
import xdg.BaseDirectory
from msm import MsmException
from mycroft.api import DeviceApi, is_paired
from mycroft.configuration import Configuration
from mycroft.util import connected
from mycroft.util.combo_lock import ComboLock
from mycroft.util.log import LOG
from .msm_wrapper import build_msm_config, create_msm
from mycroft.util.file_utils import get_temp_path
ONE_HOUR = 3600
FIVE_MINUTES = 300

def skill_is_blacklisted(skill):
    if False:
        while True:
            i = 10
    blacklist = Configuration.get()['skills']['blacklisted_skills']
    return os.path.basename(skill.path) in blacklist or skill.name in blacklist

class SkillUpdater:
    """Class facilitating skill update / install actions.

    Arguments
        bus (MessageBusClient): Optional bus emitter Used to communicate
                                with the mycroft core system and handle
                                commands.
    """
    _installed_skills_file_path = None
    _msm = None

    def __init__(self, bus=None):
        if False:
            print('Hello World!')
        self.msm_lock = ComboLock(get_temp_path('mycroft-msm.lck'))
        self.install_retries = 0
        self.config = Configuration.get()
        update_interval = self.config['skills']['update_interval']
        self.update_interval = int(update_interval) * ONE_HOUR
        self.dot_msm_path = os.path.join(self.msm.skills_dir, '.msm')
        self.next_download = self._determine_next_download_time()
        self._log_next_download_time()
        self.installed_skills = set()
        self.default_skill_install_error = False
        if bus:
            self._register_bus_handlers()

    def _register_bus_handlers(self):
        if False:
            while True:
                i = 10
        'TODO: Register bus handlers for triggering updates and such.'

    def _determine_next_download_time(self):
        if False:
            print('Hello World!')
        'Determine the initial values of the next/last download times.\n\n        Update immediately if the .msm or installed skills file is missing\n        otherwise use the timestamp on .msm as a basis.\n        '
        msm_files_exist = os.path.exists(self.dot_msm_path) and os.path.exists(self.installed_skills_file_path)
        if msm_files_exist:
            mtime = os.path.getmtime(self.dot_msm_path)
            next_download = mtime + self.update_interval
        else:
            next_download = time() - 1
        return next_download

    @property
    def installed_skills_file_path(self):
        if False:
            while True:
                i = 10
        'Property representing the path of the installed skills file.'
        if self._installed_skills_file_path is None:
            virtual_env_path = os.path.dirname(os.path.dirname(sys.executable))
            if os.access(virtual_env_path, os.W_OK | os.R_OK | os.X_OK):
                self._installed_skills_file_path = os.path.join(virtual_env_path, '.mycroft-skills')
            else:
                self._installed_skills_file_path = os.path.join(xdg.BaseDirectory.save_data_path('mycroft'), '.mycroft-skills')
        return self._installed_skills_file_path

    @property
    def msm(self):
        if False:
            for i in range(10):
                print('nop')
        if self._msm is None:
            msm_config = build_msm_config(self.config)
            self._msm = create_msm(msm_config)
        return self._msm

    @property
    def default_skill_names(self) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        'Property representing the default skills expected to be installed'
        default_skill_groups = dict(self.msm.repo.get_default_skill_names())
        default_skills = set(default_skill_groups['default'])
        platform_default_skills = default_skill_groups.get(self.msm.platform)
        if platform_default_skills is None:
            log_msg = 'No default skills found for platform {}'
            LOG.info(log_msg.format(self.msm.platform))
        else:
            default_skills.update(platform_default_skills)
        return tuple(default_skills)

    def _load_installed_skills(self):
        if False:
            print('Hello World!')
        'Load the last known skill listing from a file.'
        if os.path.isfile(self.installed_skills_file_path):
            with open(self.installed_skills_file_path) as skills_file:
                self.installed_skills = {i.strip() for i in skills_file.readlines() if i.strip()}

    def _save_installed_skills(self):
        if False:
            for i in range(10):
                print('nop')
        'Save the skill listing after the download to a file.'
        with open(self.installed_skills_file_path, 'w') as skills_file:
            for skill_name in self.installed_skills:
                skills_file.write(skill_name + '\n')

    def update_skills(self, quick=False):
        if False:
            i = 10
            return i + 15
        'Invoke MSM to install default skills and/or update installed skills\n\n        Args:\n            quick (bool): Expedite the download by running with more threads?\n        '
        LOG.info('Beginning skill update...')
        self.msm._device_skill_state = None
        success = True
        if connected():
            self._load_installed_skills()
            with self.msm_lock, self.msm.lock:
                self._apply_install_or_update(quick)
            self._save_installed_skills()
            if self.default_skill_install_error and self.install_retries < 10:
                self._schedule_retry()
                success = False
            else:
                self.install_retries = 0
                self._update_download_time()
        else:
            self.handle_not_connected()
            success = False
        if success:
            LOG.info('Skill update complete')
        return success

    def handle_not_connected(self):
        if False:
            return 10
        'Notifications of the device not being connected to the internet'
        LOG.error('msm failed, network connection not available')
        self.next_download = time() + FIVE_MINUTES

    def _apply_install_or_update(self, quick):
        if False:
            i = 10
            return i + 15
        'Invoke MSM to install or update a skill.'
        try:
            defaults = all([s.is_local for s in self.msm.default_skills.values()])
            num_threads = 20 if not defaults or quick else 2
            self.msm.apply(self.install_or_update, self.msm.list(), max_threads=num_threads)
            self.post_manifest()
        except MsmException as e:
            LOG.error('Failed to update skills: {}'.format(repr(e)))

    def post_manifest(self, reload_skills_manifest=False):
        if False:
            print('Hello World!')
        "Post the manifest of the device's skills to the backend."
        upload_allowed = self.config['skills'].get('upload_skill_manifest')
        if upload_allowed and is_paired():
            if reload_skills_manifest:
                self.msm.clear_cache()
            try:
                device_api = DeviceApi()
                device_api.upload_skills_data(self.msm.device_skill_state)
            except Exception:
                LOG.exception('Could not upload skill manifest')

    def install_or_update(self, skill):
        if False:
            print('Hello World!')
        'Install missing defaults and update existing skills'
        if self._get_device_skill_state(skill.name).get('beta', False):
            skill.sha = None
        if skill.is_local:
            skill.update()
            if skill.name not in self.installed_skills:
                skill.update_deps()
        elif skill.name in self.default_skill_names:
            try:
                self.msm.install(skill, origin='default')
            except Exception:
                if skill.name in self.default_skill_names:
                    LOG.warning('Failed to install default skill: ' + skill.name)
                    self.default_skill_install_error = True
                raise
        self.installed_skills.add(skill.name)

    def defaults_installed(self):
        if False:
            print('Hello World!')
        'Check if all default skills are installed.\n\n        Returns:\n            True if all default skills are installed, else False.\n        '
        defaults = []
        for skill in self.msm.default_skills.values():
            if not skill_is_blacklisted(skill):
                defaults.append(skill)
        return all([skill.is_local for skill in defaults])

    def _get_device_skill_state(self, skill_name):
        if False:
            print('Hello World!')
        'Get skill data structure from name.'
        device_skill_state = {}
        for msm_skill_state in self.msm.device_skill_state.get('skills', []):
            if msm_skill_state.get('name') == skill_name:
                device_skill_state = msm_skill_state
        return device_skill_state

    def _schedule_retry(self):
        if False:
            for i in range(10):
                print('nop')
        'Schedule the next skill update in the event of a failure.'
        self.install_retries += 1
        self.next_download = time() + FIVE_MINUTES
        self._log_next_download_time()
        self.default_skill_install_error = False

    def _update_download_time(self):
        if False:
            for i in range(10):
                print('nop')
        'Update timestamp on .msm file to be used when system is restarted'
        with open(self.dot_msm_path, 'a'):
            os.utime(self.dot_msm_path, None)
        self.next_download = time() + self.update_interval
        self._log_next_download_time()

    def _log_next_download_time(self):
        if False:
            for i in range(10):
                print('nop')
        LOG.info('Next scheduled skill update: ' + str(datetime.fromtimestamp(self.next_download)))