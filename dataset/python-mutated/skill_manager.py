"""Load, update and manage skills on this device."""
import os
from glob import glob
from threading import Thread, Event, Lock
from time import sleep, time, monotonic
from inspect import signature
from mycroft.api import is_paired
from mycroft.enclosure.api import EnclosureAPI
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.util.log import LOG
from .msm_wrapper import create_msm as msm_creator, build_msm_config
from .settings import SkillSettingsDownloader
from .skill_loader import SkillLoader
from .skill_updater import SkillUpdater
SKILL_MAIN_MODULE = '__init__.py'

class UploadQueue:
    """Queue for holding loaders with data that still needs to be uploaded.

    This queue can be used during startup to capture all loaders
    and then processing can be triggered at a later stage when the system is
    connected to the backend.

    After all queued settingsmeta has been processed and the queue is empty
    the queue will set the self.started flag.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._queue = []
        self.started = False
        self.lock = Lock()

    def start(self):
        if False:
            while True:
                i = 10
        'Start processing of the queue.'
        self.started = True
        self.send()

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop the queue, and hinder any further transmissions.'
        self.started = False

    def send(self):
        if False:
            while True:
                i = 10
        'Loop through all stored loaders triggering settingsmeta upload.'
        with self.lock:
            queue = self._queue
            self._queue = []
        if queue:
            LOG.info('New Settings meta to upload.')
            for loader in queue:
                if self.started:
                    loader.instance.settings_meta.upload()
                else:
                    break

    def __len__(self):
        if False:
            return 10
        return len(self._queue)

    def put(self, loader):
        if False:
            i = 10
            return i + 15
        "Append a skill loader to the queue.\n\n        If a loader is already present it's removed in favor of the new entry.\n        "
        if self.started:
            LOG.info('Updating settings meta during runtime...')
        with self.lock:
            self._queue = [e for e in self._queue if e != loader]
            self._queue.append(loader)

def _shutdown_skill(instance):
    if False:
        i = 10
        return i + 15
    'Shutdown a skill.\n\n    Call the default_shutdown method of the skill, will produce a warning if\n    the shutdown process takes longer than 1 second.\n\n    Args:\n        instance (MycroftSkill): Skill instance to shutdown\n    '
    try:
        ref_time = monotonic()
        instance.default_shutdown()
        shutdown_time = monotonic() - ref_time
        if shutdown_time > 1:
            LOG.warning('{} shutdown took {} seconds'.format(instance.skill_id, shutdown_time))
    except Exception:
        LOG.exception('Failed to shut down skill: {}'.format(instance.skill_id))

class SkillManager(Thread):
    _msm = None

    def __init__(self, bus, watchdog=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor\n\n        Args:\n            bus (event emitter): Mycroft messagebus connection\n            watchdog (callable): optional watchdog function\n        '
        super(SkillManager, self).__init__()
        self.bus = bus
        self._watchdog = watchdog or (lambda : None)
        self._stop_event = Event()
        self._connected_event = Event()
        self.config = Configuration.get()
        self.upload_queue = UploadQueue()
        self.skill_loaders = {}
        self.enclosure = EnclosureAPI(bus)
        self.initial_load_complete = False
        self.num_install_retries = 0
        self.settings_downloader = SkillSettingsDownloader(self.bus)
        self.empty_skill_dirs = set()
        self._alive_status = False
        self._loaded_status = False
        self.skill_updater = SkillUpdater()
        self._define_message_bus_events()
        self.daemon = True

    def _define_message_bus_events(self):
        if False:
            while True:
                i = 10
        'Define message bus events with handlers defined in this class.'
        self.bus.on('skill.converse.request', self.handle_converse_request)
        self.bus.on('mycroft.internet.connected', lambda x: self._connected_event.set())
        self.bus.on('skillmanager.update', self.schedule_now)
        self.bus.on('skillmanager.list', self.send_skill_list)
        self.bus.on('skillmanager.deactivate', self.deactivate_skill)
        self.bus.on('skillmanager.keep', self.deactivate_except)
        self.bus.on('skillmanager.activate', self.activate_skill)
        self.bus.on('mycroft.paired', self.handle_paired)
        self.bus.on('mycroft.skills.settings.update', self.settings_downloader.download)

    @property
    def skills_config(self):
        if False:
            i = 10
            return i + 15
        return self.config['skills']

    @property
    def msm(self):
        if False:
            print('Hello World!')
        if self._msm is None:
            msm_config = build_msm_config(self.config)
            self._msm = msm_creator(msm_config)
        return self._msm

    @staticmethod
    def create_msm():
        if False:
            i = 10
            return i + 15
        LOG.debug('instantiating msm via static method...')
        msm_config = build_msm_config(Configuration.get())
        msm_instance = msm_creator(msm_config)
        return msm_instance

    def schedule_now(self, _):
        if False:
            i = 10
            return i + 15
        self.skill_updater.next_download = time() - 1

    def _start_settings_update(self):
        if False:
            while True:
                i = 10
        LOG.info('Start settings update')
        self.skill_updater.post_manifest(reload_skills_manifest=True)
        self.upload_queue.start()
        LOG.info('All settings meta has been processed or upload has started')
        self.settings_downloader.download()
        LOG.info('Skill settings downloading has started')

    def handle_paired(self, _):
        if False:
            while True:
                i = 10
        'Trigger upload of skills manifest after pairing.'
        self._start_settings_update()

    def load_priority(self):
        if False:
            print('Hello World!')
        skills = {skill.name: skill for skill in self.msm.all_skills}
        priority_skills = self.skills_config.get('priority_skills', [])
        for skill_name in priority_skills:
            skill = skills.get(skill_name)
            if skill is not None:
                if not skill.is_local:
                    try:
                        self.msm.install(skill)
                    except Exception:
                        log_msg = 'Downloading priority skill: {} failed'
                        LOG.exception(log_msg.format(skill_name))
                        continue
                loader = self._load_skill(skill.path)
                if loader:
                    self.upload_queue.put(loader)
            else:
                LOG.error("Priority skill {} can't be found".format(skill_name))
        self._alive_status = True

    def run(self):
        if False:
            i = 10
            return i + 15
        'Load skills and update periodically from disk and internet.'
        self._remove_git_locks()
        self._connected_event.wait()
        if not self.skill_updater.defaults_installed() and self.skills_config['auto_update']:
            LOG.info('Not all default skills are installed, performing skill update...')
            self.skill_updater.update_skills()
        self._load_on_startup()
        if is_paired() and (not self.upload_queue.started):
            self._start_settings_update()
        while not self._stop_event.is_set():
            try:
                self._unload_removed_skills()
                self._reload_modified_skills()
                self._load_new_skills()
                self._update_skills()
                if is_paired() and self.upload_queue.started and (len(self.upload_queue) > 0):
                    self.msm.clear_cache()
                    self.skill_updater.post_manifest()
                    self.upload_queue.send()
                self._watchdog()
                sleep(2)
            except Exception:
                LOG.exception('Something really unexpected has occured and the skill manager loop safety harness was hit.')
                sleep(30)

    def _remove_git_locks(self):
        if False:
            print('Hello World!')
        'If git gets killed from an abrupt shutdown it leaves lock files.'
        for i in glob(os.path.join(self.msm.skills_dir, '*/.git/index.lock')):
            LOG.warning('Found and removed git lock file: ' + i)
            os.remove(i)

    def _load_on_startup(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle initial skill load.'
        LOG.info('Loading installed skills...')
        self._load_new_skills()
        LOG.info('Skills all loaded!')
        self.bus.emit(Message('mycroft.skills.initialized'))
        self._loaded_status = True

    def _reload_modified_skills(self):
        if False:
            print('Hello World!')
        'Handle reload of recently changed skill(s)'
        for skill_dir in self._get_skill_directories():
            try:
                skill_loader = self.skill_loaders.get(skill_dir)
                if skill_loader is not None and skill_loader.reload_needed():
                    if skill_loader.reload():
                        self.upload_queue.put(skill_loader)
            except Exception:
                LOG.exception('Unhandled exception occured while reloading {}'.format(skill_dir))

    def _load_new_skills(self):
        if False:
            print('Hello World!')
        'Handle load of skills installed since startup.'
        for skill_dir in self._get_skill_directories():
            if skill_dir not in self.skill_loaders:
                loader = self._load_skill(skill_dir)
                if loader:
                    self.upload_queue.put(loader)

    def _load_skill(self, skill_directory):
        if False:
            while True:
                i = 10
        skill_loader = SkillLoader(self.bus, skill_directory)
        try:
            load_status = skill_loader.load()
        except Exception:
            LOG.exception('Load of skill {} failed!'.format(skill_directory))
            load_status = False
        finally:
            self.skill_loaders[skill_directory] = skill_loader
        return skill_loader if load_status else None

    def _get_skill_directories(self):
        if False:
            i = 10
            return i + 15
        skill_glob = glob(os.path.join(self.msm.skills_dir, '*/'))
        skill_directories = []
        for skill_dir in skill_glob:
            if SKILL_MAIN_MODULE in os.listdir(skill_dir):
                skill_directories.append(skill_dir.rstrip('/'))
                if skill_dir in self.empty_skill_dirs:
                    self.empty_skill_dirs.discard(skill_dir)
            elif skill_dir not in self.empty_skill_dirs:
                self.empty_skill_dirs.add(skill_dir)
                LOG.debug('Found skills directory with no skill: ' + skill_dir)
        return skill_directories

    def _unload_removed_skills(self):
        if False:
            i = 10
            return i + 15
        'Shutdown removed skills.'
        skill_dirs = self._get_skill_directories()
        removed_skills = [s for s in self.skill_loaders.keys() if s not in skill_dirs]
        for skill_dir in removed_skills:
            skill = self.skill_loaders[skill_dir]
            LOG.info('removing {}'.format(skill.skill_id))
            try:
                skill.unload()
            except Exception:
                LOG.exception('Failed to shutdown skill ' + skill.id)
            del self.skill_loaders[skill_dir]
        if removed_skills:
            self.skill_updater.post_manifest(reload_skills_manifest=True)

    def _update_skills(self):
        if False:
            return 10
        'Update skills once an hour if update is enabled'
        do_skill_update = time() >= self.skill_updater.next_download and self.skills_config['auto_update']
        if do_skill_update:
            self.skill_updater.update_skills()

    def is_alive(self, message=None):
        if False:
            for i in range(10):
                print('nop')
        'Respond to is_alive status request.'
        return self._alive_status

    def is_all_loaded(self, message=None):
        if False:
            while True:
                i = 10
        ' Respond to all_loaded status request.'
        return self._loaded_status

    def send_skill_list(self, _):
        if False:
            while True:
                i = 10
        'Send list of loaded skills.'
        try:
            message_data = {}
            for (skill_dir, skill_loader) in self.skill_loaders.items():
                message_data[skill_loader.skill_id] = dict(active=skill_loader.active and skill_loader.loaded, id=skill_loader.skill_id)
            self.bus.emit(Message('mycroft.skills.list', data=message_data))
        except Exception:
            LOG.exception('Failed to send skill list')

    def deactivate_skill(self, message):
        if False:
            while True:
                i = 10
        'Deactivate a skill.'
        try:
            for skill_loader in self.skill_loaders.values():
                if message.data['skill'] == skill_loader.skill_id:
                    skill_loader.deactivate()
        except Exception:
            LOG.exception('Failed to deactivate ' + message.data['skill'])

    def deactivate_except(self, message):
        if False:
            return 10
        'Deactivate all skills except the provided.'
        try:
            skill_to_keep = message.data['skill']
            LOG.info('Deactivating all skills except {}'.format(skill_to_keep))
            loaded_skill_file_names = [os.path.basename(skill_dir) for skill_dir in self.skill_loaders]
            if skill_to_keep in loaded_skill_file_names:
                for skill in self.skill_loaders.values():
                    if skill.skill_id != skill_to_keep:
                        skill.deactivate()
            else:
                LOG.info("Couldn't find skill " + message.data['skill'])
        except Exception:
            LOG.exception('An error occurred during skill deactivation!')

    def activate_skill(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Activate a deactivated skill.'
        try:
            for skill_loader in self.skill_loaders.values():
                if message.data['skill'] in ('all', skill_loader.skill_id) and (not skill_loader.active):
                    skill_loader.activate()
        except Exception:
            LOG.exception("Couldn't activate skill")

    def stop(self):
        if False:
            return 10
        'Tell the manager to shutdown.'
        self._stop_event.set()
        self.settings_downloader.stop_downloading()
        self.upload_queue.stop()
        for skill_loader in self.skill_loaders.values():
            if skill_loader.instance is not None:
                _shutdown_skill(skill_loader.instance)

    def handle_converse_request(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Check if the targeted skill id can handle conversation\n\n        If supported, the conversation is invoked.\n        '
        skill_id = message.data['skill_id']
        skill_found = False
        for skill_loader in self.skill_loaders.values():
            if skill_loader.skill_id == skill_id:
                skill_found = True
                if not skill_loader.loaded:
                    error_message = 'converse requested but skill not loaded'
                    self._emit_converse_error(message, skill_id, error_message)
                    break
                try:
                    if len(signature(skill_loader.instance.converse).parameters) == 1:
                        result = skill_loader.instance.converse(message=message)
                    else:
                        utterances = message.data['utterances']
                        lang = message.data['lang']
                        result = skill_loader.instance.converse(utterances=utterances, lang=lang)
                    self._emit_converse_response(result, message, skill_loader)
                except Exception:
                    error_message = 'exception in converse method'
                    LOG.exception(error_message)
                    self._emit_converse_error(message, skill_id, error_message)
                finally:
                    break
        if not skill_found:
            error_message = 'skill id does not exist'
            self._emit_converse_error(message, skill_id, error_message)

    def _emit_converse_error(self, message, skill_id, error_msg):
        if False:
            return 10
        'Emit a message reporting the error back to the intent service.'
        reply = message.reply('skill.converse.response', data=dict(skill_id=skill_id, error=error_msg))
        self.bus.emit(reply)

    def _emit_converse_response(self, result, message, skill_loader):
        if False:
            i = 10
            return i + 15
        reply = message.reply('skill.converse.response', data=dict(skill_id=skill_loader.skill_id, result=result))
        self.bus.emit(reply)