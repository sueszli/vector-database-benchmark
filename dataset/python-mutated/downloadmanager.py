"""Youtubedlg module for managing the download process.

This module is responsible for managing the download process
and update the GUI interface.

Attributes:
    MANAGER_PUB_TOPIC (string): wxPublisher subscription topic of the
        DownloadManager thread.

    WORKER_PUB_TOPIC (string): wxPublisher subscription topic of the
        Worker thread.

Note:
    It's not the actual module that downloads the urls
    thats the job of the 'downloaders' module.

"""
from __future__ import unicode_literals
import time
import os.path
from threading import Thread, RLock, Lock
from wx import CallAfter
from wx.lib.pubsub import setuparg1
from wx.lib.pubsub import pub as Publisher
from .parsers import OptionsParser
from .updatemanager import UpdateThread
from .downloaders import YoutubeDLDownloader
from .utils import YOUTUBEDL_BIN, os_path_exists, format_bytes, to_string, to_bytes
MANAGER_PUB_TOPIC = 'dlmanager'
WORKER_PUB_TOPIC = 'dlworker'
_SYNC_LOCK = RLock()

def synchronized(lock):
    if False:
        print('Hello World!')

    def _decorator(func):
        if False:
            return 10

        def _wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            lock.acquire()
            ret_value = func(*args, **kwargs)
            lock.release()
            return ret_value
        return _wrapper
    return _decorator

class DownloadItem(object):
    """Object that represents a download.

    Attributes:
        STAGES (tuple): Main stages of the download item.

        ACTIVE_STAGES (tuple): Sub stages of the 'Active' stage.

        COMPLETED_STAGES (tuple): Sub stages of the 'Completed' stage.

        ERROR_STAGES (tuple): Sub stages of the 'Error' stage.

    Args:
        url (string): URL that corresponds to the download item.

        options (list): Options list to use during the download phase.

    """
    STAGES = ('Queued', 'Active', 'Paused', 'Completed', 'Error')
    ACTIVE_STAGES = ('Pre Processing', 'Downloading', 'Post Processing')
    COMPLETED_STAGES = ('Finished', 'Warning', 'Already Downloaded')
    ERROR_STAGES = ('Error', 'Stopped', 'Filesize Abort')

    def __init__(self, url, options):
        if False:
            print('Hello World!')
        self.url = url
        self.options = options
        self.object_id = hash(url + to_string(options))
        self.reset()

    @property
    def stage(self):
        if False:
            while True:
                i = 10
        return self._stage

    @stage.setter
    def stage(self, value):
        if False:
            print('Hello World!')
        if value not in self.STAGES:
            raise ValueError(value)
        if value == 'Queued':
            self.progress_stats['status'] = value
        if value == 'Active':
            self.progress_stats['status'] = self.ACTIVE_STAGES[0]
        if value == 'Completed':
            self.progress_stats['status'] = self.COMPLETED_STAGES[0]
        if value == 'Paused':
            self.progress_stats['status'] = value
        if value == 'Error':
            self.progress_stats['status'] = self.ERROR_STAGES[0]
        self._stage = value

    def reset(self):
        if False:
            return 10
        if hasattr(self, '_stage') and self._stage == self.STAGES[1]:
            raise RuntimeError("Cannot reset an 'Active' item")
        self._stage = self.STAGES[0]
        self.path = ''
        self.filenames = []
        self.extensions = []
        self.filesizes = []
        self.default_values = {'filename': self.url, 'extension': '-', 'filesize': '-', 'percent': '0%', 'speed': '-', 'eta': '-', 'status': self.stage, 'playlist_size': '', 'playlist_index': ''}
        self.progress_stats = dict(self.default_values)
        self.playlist_index_changed = False

    def get_files(self):
        if False:
            while True:
                i = 10
        'Returns a list that contains all the system files bind to this object.'
        files = []
        for (index, item) in enumerate(self.filenames):
            filename = item + self.extensions[index]
            files.append(os.path.join(self.path, filename))
        return files

    def update_stats(self, stats_dict):
        if False:
            return 10
        'Updates the progress_stats dict from the given dictionary.'
        assert isinstance(stats_dict, dict)
        for key in stats_dict:
            if key in self.progress_stats:
                value = stats_dict[key]
                if not isinstance(value, basestring) or not value:
                    self.progress_stats[key] = self.default_values[key]
                else:
                    self.progress_stats[key] = value
        if 'playlist_index' in stats_dict:
            self.playlist_index_changed = True
        if 'filename' in stats_dict:
            if self.playlist_index_changed:
                self.filenames = []
                self.extensions = []
                self.filesizes = []
                self.playlist_index_changed = False
            self.filenames.append(stats_dict['filename'])
        if 'extension' in stats_dict:
            self.extensions.append(stats_dict['extension'])
        if 'path' in stats_dict:
            self.path = stats_dict['path']
        if 'filesize' in stats_dict:
            if stats_dict['percent'] == '100%' and len(self.filesizes) < len(self.filenames):
                filesize = stats_dict['filesize'].lstrip('~')
                self.filesizes.append(to_bytes(filesize))
        if 'status' in stats_dict:
            if stats_dict['status'] == self.ACTIVE_STAGES[2] and len(self.filesizes) == 2:
                post_proc_filesize = self.filesizes[0] + self.filesizes[1]
                self.filesizes.append(post_proc_filesize)
                self.progress_stats['filesize'] = format_bytes(post_proc_filesize)
            self._set_stage(stats_dict['status'])

    def _set_stage(self, status):
        if False:
            print('Hello World!')
        if status in self.ACTIVE_STAGES:
            self._stage = self.STAGES[1]
        if status in self.COMPLETED_STAGES:
            self._stage = self.STAGES[3]
        if status in self.ERROR_STAGES:
            self._stage = self.STAGES[4]

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.object_id == other.object_id

class DownloadList(object):
    """List like data structure that contains DownloadItems.

    Args:
        items (list): List that contains DownloadItems.

    """

    def __init__(self, items=None):
        if False:
            return 10
        assert isinstance(items, list) or items is None
        if items is None:
            self._items_dict = {}
            self._items_list = []
        else:
            self._items_list = [item.object_id for item in items]
            self._items_dict = {item.object_id: item for item in items}

    @synchronized(_SYNC_LOCK)
    def clear(self):
        if False:
            print('Hello World!')
        "Removes all the items from the list even the 'Active' ones."
        self._items_list = []
        self._items_dict = {}

    @synchronized(_SYNC_LOCK)
    def insert(self, item):
        if False:
            i = 10
            return i + 15
        'Inserts the given item to the list. Does not check for duplicates. '
        self._items_list.append(item.object_id)
        self._items_dict[item.object_id] = item

    @synchronized(_SYNC_LOCK)
    def remove(self, object_id):
        if False:
            while True:
                i = 10
        "Removes an item from the list.\n\n        Removes the item with the corresponding object_id from\n        the list if the item is not in 'Active' state.\n\n        Returns:\n            True on success else False.\n\n        "
        if self._items_dict[object_id].stage != 'Active':
            self._items_list.remove(object_id)
            del self._items_dict[object_id]
            return True
        return False

    @synchronized(_SYNC_LOCK)
    def fetch_next(self):
        if False:
            i = 10
            return i + 15
        'Returns the next queued item on the list.\n\n        Returns:\n            Next queued item or None if no other item exist.\n\n        '
        for object_id in self._items_list:
            cur_item = self._items_dict[object_id]
            if cur_item.stage == 'Queued':
                return cur_item
        return None

    @synchronized(_SYNC_LOCK)
    def move_up(self, object_id):
        if False:
            return 10
        'Moves the item with the corresponding object_id up to the list.'
        index = self._items_list.index(object_id)
        if index > 0:
            self._swap(index, index - 1)
            return True
        return False

    @synchronized(_SYNC_LOCK)
    def move_down(self, object_id):
        if False:
            i = 10
            return i + 15
        'Moves the item with the corresponding object_id down to the list.'
        index = self._items_list.index(object_id)
        if index < len(self._items_list) - 1:
            self._swap(index, index + 1)
            return True
        return False

    @synchronized(_SYNC_LOCK)
    def get_item(self, object_id):
        if False:
            for i in range(10):
                print('nop')
        'Returns the DownloadItem with the given object_id.'
        return self._items_dict[object_id]

    @synchronized(_SYNC_LOCK)
    def has_item(self, object_id):
        if False:
            i = 10
            return i + 15
        'Returns True if the given object_id is in the list else False.'
        return object_id in self._items_list

    @synchronized(_SYNC_LOCK)
    def get_items(self):
        if False:
            i = 10
            return i + 15
        'Returns a list with all the items.'
        return [self._items_dict[object_id] for object_id in self._items_list]

    @synchronized(_SYNC_LOCK)
    def change_stage(self, object_id, new_stage):
        if False:
            for i in range(10):
                print('nop')
        'Change the stage of the item with the given object_id.'
        self._items_dict[object_id].stage = new_stage

    @synchronized(_SYNC_LOCK)
    def index(self, object_id):
        if False:
            i = 10
            return i + 15
        'Get the zero based index of the item with the given object_id.'
        if object_id in self._items_list:
            return self._items_list.index(object_id)
        return -1

    @synchronized(_SYNC_LOCK)
    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._items_list)

    def _swap(self, index1, index2):
        if False:
            print('Hello World!')
        (self._items_list[index1], self._items_list[index2]) = (self._items_list[index2], self._items_list[index1])

class DownloadManager(Thread):
    """Manages the download process.

    Attributes:
        WAIT_TIME (float): Time in seconds to sleep.

    Args:
        download_list (DownloadList): List that contains items to download.

        opt_manager (optionsmanager.OptionsManager): Object responsible for
            managing the youtubedlg options.

        log_manager (logmanager.LogManager): Object responsible for writing
            errors to the log.

    """
    WAIT_TIME = 0.1

    def __init__(self, parent, download_list, opt_manager, log_manager=None):
        if False:
            i = 10
            return i + 15
        super(DownloadManager, self).__init__()
        self.parent = parent
        self.opt_manager = opt_manager
        self.log_manager = log_manager
        self.download_list = download_list
        self._time_it_took = 0
        self._successful = 0
        self._running = True
        log_lock = None if log_manager is None else Lock()
        wparams = (opt_manager, self._youtubedl_path(), log_manager, log_lock)
        self._workers = [Worker(*wparams) for _ in xrange(opt_manager.options['workers_number'])]
        self.start()

    @property
    def successful(self):
        if False:
            print('Hello World!')
        'Returns number of successful downloads. '
        return self._successful

    @property
    def time_it_took(self):
        if False:
            print('Hello World!')
        'Returns time(seconds) it took for the download process\n        to complete. '
        return self._time_it_took

    def run(self):
        if False:
            i = 10
            return i + 15
        if not self.opt_manager.options['disable_update']:
            self._check_youtubedl()
        self._time_it_took = time.time()
        while self._running:
            item = self.download_list.fetch_next()
            if item is not None:
                worker = self._get_worker()
                if worker is not None:
                    worker.download(item.url, item.options, item.object_id)
                    self.download_list.change_stage(item.object_id, 'Active')
            if item is None and self._jobs_done():
                break
            time.sleep(self.WAIT_TIME)
        for worker in self._workers:
            worker.close()
        for worker in self._workers:
            worker.join()
            self._successful += worker.successful
        self._time_it_took = time.time() - self._time_it_took
        if not self._running:
            self._talk_to_gui('closed')
        else:
            self._talk_to_gui('finished')

    def active(self):
        if False:
            return 10
        'Returns number of active items.\n\n        Note:\n            active_items = (workers that work) + (items waiting in the url_list).\n\n        '
        return len(self.download_list)

    def stop_downloads(self):
        if False:
            return 10
        "Stop the download process. Also send 'closing'\n        signal back to the GUI.\n\n        Note:\n            It does NOT kill the workers thats the job of the\n            clean up task in the run() method.\n\n        "
        self._talk_to_gui('closing')
        self._running = False

    def add_url(self, url):
        if False:
            return 10
        'Add given url to the download_list.\n\n        Args:\n            url (dict): Python dictionary that contains two keys.\n                The url and the index of the corresponding row in which\n                the worker should send back the information about the\n                download process.\n\n        '
        self.download_list.append(url)

    def send_to_worker(self, data):
        if False:
            for i in range(10):
                print('nop')
        "Send data to the Workers.\n\n        Args:\n            data (dict): Python dictionary that holds the 'index'\n            which is used to identify the Worker thread and the data which\n            can be any of the Worker's class valid data. For a list of valid\n            data keys see __init__() under the Worker class.\n\n        "
        if 'index' in data:
            for worker in self._workers:
                if worker.has_index(data['index']):
                    worker.update_data(data)

    def _talk_to_gui(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Send data back to the GUI using wxCallAfter and wxPublisher.\n\n        Args:\n            data (string): Unique signal string that informs the GUI for the\n                download process.\n\n        Note:\n            DownloadManager supports 4 signals.\n                1) closing: The download process is closing.\n                2) closed: The download process has closed.\n                3) finished: The download process was completed normally.\n                4) report_active: Signal the gui to read the number of active\n                    downloads using the active() method.\n\n        '
        CallAfter(Publisher.sendMessage, MANAGER_PUB_TOPIC, data)

    def _check_youtubedl(self):
        if False:
            return 10
        'Check if youtube-dl binary exists. If not try to download it. '
        if not os_path_exists(self._youtubedl_path()) and self.parent.update_thread is None:
            self.parent.update_thread = UpdateThread(self.opt_manager.options['youtubedl_path'], True)
            self.parent.update_thread.join()
            self.parent.update_thread = None

    def _get_worker(self):
        if False:
            for i in range(10):
                print('nop')
        for worker in self._workers:
            if worker.available():
                return worker
        return None

    def _jobs_done(self):
        if False:
            print('Hello World!')
        'Returns True if the workers have finished their jobs else False. '
        for worker in self._workers:
            if not worker.available():
                return False
        return True

    def _youtubedl_path(self):
        if False:
            return 10
        'Returns the path to youtube-dl binary. '
        path = self.opt_manager.options['youtubedl_path']
        path = os.path.join(path, YOUTUBEDL_BIN)
        return path

class Worker(Thread):
    """Simple worker which downloads the given url using a downloader
    from the downloaders.py module.

    Attributes:
        WAIT_TIME (float): Time in seconds to sleep.

    Args:
        opt_manager (optionsmanager.OptionsManager): Check DownloadManager
            description.

        youtubedl (string): Absolute path to youtube-dl binary.

        log_manager (logmanager.LogManager): Check DownloadManager
            description.

        log_lock (threading.Lock): Synchronization lock for the log_manager.
            If the log_manager is set (not None) then the caller has to make
            sure that the log_lock is also set.

    Note:
        For available data keys see self._data under the __init__() method.

    """
    WAIT_TIME = 0.1

    def __init__(self, opt_manager, youtubedl, log_manager=None, log_lock=None):
        if False:
            for i in range(10):
                print('nop')
        super(Worker, self).__init__()
        self.opt_manager = opt_manager
        self.log_manager = log_manager
        self.log_lock = log_lock
        self._downloader = YoutubeDLDownloader(youtubedl, self._data_hook, self._log_data)
        self._options_parser = OptionsParser()
        self._successful = 0
        self._running = True
        self._options = None
        self._wait_for_reply = False
        self._data = {'playlist_index': None, 'playlist_size': None, 'new_filename': None, 'extension': None, 'filesize': None, 'filename': None, 'percent': None, 'status': None, 'index': None, 'speed': None, 'path': None, 'eta': None, 'url': None}
        self.start()

    def run(self):
        if False:
            while True:
                i = 10
        while self._running:
            if self._data['url'] is not None:
                ret_code = self._downloader.download(self._data['url'], self._options)
                if ret_code == YoutubeDLDownloader.OK or ret_code == YoutubeDLDownloader.ALREADY or ret_code == YoutubeDLDownloader.WARNING:
                    self._successful += 1
                self._reset()
            time.sleep(self.WAIT_TIME)
        self._downloader.close()

    def download(self, url, options, object_id):
        if False:
            while True:
                i = 10
        'Download given item.\n\n        Args:\n            item (dict): Python dictionary that contains two keys.\n                The url and the index of the corresponding row in which\n                the worker should send back the information about the\n                download process.\n\n        '
        self._data['url'] = url
        self._options = options
        self._data['index'] = object_id

    def stop_download(self):
        if False:
            i = 10
            return i + 15
        'Stop the download process of the worker. '
        self._downloader.stop()

    def close(self):
        if False:
            print('Hello World!')
        'Kill the worker after stopping the download process. '
        self._running = False
        self._downloader.stop()

    def available(self):
        if False:
            i = 10
            return i + 15
        'Return True if the worker has no job else False. '
        return self._data['url'] is None

    def has_index(self, index):
        if False:
            print('Hello World!')
        "Return True if index is equal to self._data['index'] else False. "
        return self._data['index'] == index

    def update_data(self, data):
        if False:
            return 10
        'Update self._data from the given data. '
        if self._wait_for_reply:
            for key in data:
                self._data[key] = data[key]
            self._wait_for_reply = False

    @property
    def successful(self):
        if False:
            return 10
        'Return the number of successful downloads for current worker. '
        return self._successful

    def _reset(self):
        if False:
            return 10
        'Reset self._data back to the original state. '
        for key in self._data:
            self._data[key] = None

    def _log_data(self, data):
        if False:
            print('Hello World!')
        'Callback method for self._downloader.\n\n        This method is used to write the given data in a synchronized way\n        to the log file using the self.log_manager and the self.log_lock.\n\n        Args:\n            data (string): String to write to the log file.\n\n        '
        if self.log_manager is not None:
            self.log_lock.acquire()
            self.log_manager.log(data)
            self.log_lock.release()

    def _data_hook(self, data):
        if False:
            i = 10
            return i + 15
        'Callback method for self._downloader.\n\n        This method updates self._data and sends the updates back to the\n        GUI using the self._talk_to_gui() method.\n\n        Args:\n            data (dict): Python dictionary which contains information\n                about the download process. For more info see the\n                extract_data() function under the downloaders.py module.\n\n        '
        self._talk_to_gui('send', data)

    def _talk_to_gui(self, signal, data):
        if False:
            i = 10
            return i + 15
        "Communicate with the GUI using wxCallAfter and wxPublisher.\n\n        Send/Ask data to/from the GUI. Note that if the signal is 'receive'\n        then the Worker will wait until it receives a reply from the GUI.\n\n        Args:\n            signal (string): Unique string that informs the GUI about the\n                communication procedure.\n\n            data (dict): Python dictionary which holds the data to be sent\n                back to the GUI. If the signal is 'send' then the dictionary\n                contains the updates for the GUI (e.g. percentage, eta). If\n                the signal is 'receive' then the dictionary contains exactly\n                three keys. The 'index' (row) from which we want to retrieve\n                the data, the 'source' which identifies a column in the\n                wxListCtrl widget and the 'dest' which tells the wxListCtrl\n                under which key to store the retrieved data.\n\n        Note:\n            Worker class supports 2 signals.\n                1) send: The Worker sends data back to the GUI\n                         (e.g. Send status updates).\n                2) receive: The Worker asks data from the GUI\n                            (e.g. Receive the name of a file).\n\n        Structure:\n            ('send', {'index': <item_row>, data_to_send*})\n\n            ('receive', {'index': <item_row>, 'source': 'source_key', 'dest': 'destination_key'})\n\n        "
        data['index'] = self._data['index']
        if signal == 'receive':
            self._wait_for_reply = True
        CallAfter(Publisher.sendMessage, WORKER_PUB_TOPIC, (signal, data))