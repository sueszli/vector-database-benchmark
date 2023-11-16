import json
import errno
import os
import time
import traceback
from cme.protocols.smb.remotefile import RemoteFile
from impacket.smb3structs import FILE_READ_DATA
from impacket.smbconnection import SessionError
CHUNK_SIZE = 4096

def human_size(nbytes):
    if False:
        print('Hello World!')
    '\n    This function takes a number of bytes as input and converts it to a human-readable\n    size representation with appropriate units (e.g., KB, MB, GB, TB).\n    '
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    for i in range(len(suffixes)):
        if nbytes < 1024 or i == len(suffixes) - 1:
            break
        nbytes /= 1024.0
    size_str = f'{nbytes:.2f}'.rstrip('0').rstrip('.')
    return f'{size_str} {suffixes[i]}'

def human_time(timestamp):
    if False:
        print('Hello World!')
    'This function takes a numerical timestamp (seconds since the epoch) and formats it\n    as a human-readable date and time in the format "YYYY-MM-DD HH:MM:SS".\n    '
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

def make_dirs(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function attempts to create directories at the given path. It handles the\n    exception `os.errno.EEXIST` that may occur if the directories already exist.\n    '
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass

def get_list_from_option(opt):
    if False:
        return 10
    '\n    This function takes a comma-separated string and converts it to a list of lowercase strings.\n    It filters out empty strings from the input before converting.\n    '
    return list(map(lambda o: o.lower(), filter(bool, opt.split(','))))

class SMBSpiderPlus:

    def __init__(self, smb, logger, download_flag, stats_flag, exclude_exts, exclude_filter, max_file_size, output_folder):
        if False:
            print('Hello World!')
        self.smb = smb
        self.host = self.smb.conn.getRemoteHost()
        self.max_connection_attempts = 5
        self.logger = logger
        self.results = {}
        self.stats = {'shares': list(), 'shares_readable': list(), 'shares_writable': list(), 'num_shares_filtered': 0, 'num_folders': 0, 'num_folders_filtered': 0, 'num_files': 0, 'file_sizes': list(), 'file_exts': set(), 'num_get_success': 0, 'num_get_fail': 0, 'num_files_filtered': 0, 'num_files_unmodified': 0, 'num_files_updated': 0}
        self.download_flag = download_flag
        self.stats_flag = stats_flag
        self.exclude_filter = exclude_filter
        self.exclude_exts = exclude_exts
        self.max_file_size = max_file_size
        self.output_folder = output_folder
        make_dirs(self.output_folder)

    def reconnect(self):
        if False:
            return 10
        'This function performs a series of reconnection attempts, up to `self.max_connection_attempts`,\n        with a 3-second delay between each attempt. It renegotiates the session by creating a new\n        connection object and logging in again.\n        '
        for i in range(1, self.max_connection_attempts + 1):
            self.logger.display(f'Reconnection attempt #{i}/{self.max_connection_attempts} to server.')
            time.sleep(3)
            self.smb.create_conn_obj()
            self.smb.login()
            return True
        return False

    def list_path(self, share, subfolder):
        if False:
            print('Hello World!')
        'This function returns a list of paths for a given share/folder.'
        filelist = []
        try:
            filelist = self.smb.conn.listPath(share, subfolder + '*')
        except SessionError as e:
            self.logger.debug(f'Failed listing files on share "{share}" in folder "{subfolder}".')
            self.logger.debug(str(e))
            if 'STATUS_ACCESS_DENIED' in str(e):
                self.logger.debug(f'Cannot list files in folder "{subfolder}".')
            elif 'STATUS_OBJECT_PATH_NOT_FOUND' in str(e):
                self.logger.debug(f'The folder {subfolder} does not exist.')
            elif self.reconnect():
                filelist = self.list_path(share, subfolder)
        return filelist

    def get_remote_file(self, share, path):
        if False:
            return 10
        'This function will check if a path is readable in a SMB share.'
        try:
            remote_file = RemoteFile(self.smb.conn, path, share, access=FILE_READ_DATA)
            return remote_file
        except SessionError:
            if self.reconnect():
                return self.get_remote_file(share, path)
            return None

    def read_chunk(self, remote_file, chunk_size=CHUNK_SIZE):
        if False:
            for i in range(10):
                print('nop')
        'This function reads the next chunk of data from the provided remote file using\n        the specified chunk size. If a `SessionError` is encountered,\n        it retries up to 3 times by reconnecting the SMB connection. If the maximum number\n        of retries is exhausted or an unexpected exception occurs, it returns an empty chunk.\n        '
        chunk = ''
        retry = 3
        while retry > 0:
            retry -= 1
            try:
                chunk = remote_file.read(chunk_size)
                break
            except SessionError:
                if self.reconnect():
                    remote_file.__smbConnection = self.smb.conn
                    return self.read_chunk(remote_file)
            except Exception:
                traceback.print_exc()
                break
        return chunk

    def get_file_save_path(self, remote_file):
        if False:
            while True:
                i = 10
        'This function processes the remote file path to extract the filename and the folder\n        path where the file should be saved locally. It converts forward slashes (/) and backslashes (\\)\n        in the remote file path to the appropriate path separator for the local file system.\n        The folder path and filename are then obtained separately.\n        '
        remote_file_path = str(remote_file)[2:].replace('/', os.path.sep).replace('\\', os.path.sep)
        (folder, filename) = os.path.split(remote_file_path)
        folder = os.path.join(self.output_folder, folder)
        return (folder, filename)

    def spider_shares(self):
        if False:
            i = 10
            return i + 15
        'This function enumerates all available shares for the SMB connection, spiders\n        through the readable shares, and saves the metadata of the shares to a JSON file.\n        '
        self.logger.info('Enumerating shares for spidering.')
        shares = self.smb.shares()
        try:
            for share in shares:
                share_perms = share['access']
                share_name = share['name']
                self.stats['shares'].append(share_name)
                self.logger.info(f'Share "{share_name}" has perms {share_perms}')
                if 'WRITE' in share_perms:
                    self.stats['shares_writable'].append(share_name)
                if 'READ' in share_perms:
                    self.stats['shares_readable'].append(share_name)
                else:
                    self.logger.debug(f'Share "{share_name}" not readable.')
                    continue
                if share_name.lower() in self.exclude_filter:
                    self.logger.info(f'Share "{share_name}" has been excluded.')
                    self.stats['num_shares_filtered'] += 1
                    continue
                try:
                    self.results[share_name] = {}
                    self.spider_folder(share_name, '')
                except SessionError:
                    traceback.print_exc()
                    self.logger.fail(f'Got a session error while spidering.')
                    self.reconnect()
        except Exception as e:
            traceback.print_exc()
            self.logger.fail(f'Error enumerating shares: {str(e)}')
        self.dump_folder_metadata(self.results)
        if self.stats_flag:
            self.print_stats()
        return self.results

    def spider_folder(self, share_name, folder):
        if False:
            i = 10
            return i + 15
        'This recursive function traverses through the contents of the specified share and folder.\n        It checks each entry (file or folder) against various filters, performs file metadata recording,\n        and downloads eligible files if the download flag is set.\n        '
        self.logger.info(f'Spider share "{share_name}" in folder "{folder}".')
        filelist = self.list_path(share_name, folder + '*')
        for result in filelist:
            next_filedir = result.get_longname()
            if next_filedir in ['.', '..']:
                continue
            next_fullpath = folder + next_filedir
            result_type = 'folder' if result.is_directory() else 'file'
            self.stats[f'num_{result_type}s'] += 1
            if any((d in next_filedir.lower() for d in self.exclude_filter)):
                self.logger.info(f'The {result_type} "{next_filedir}" has been excluded')
                self.stats[f'{result_type}s_filtered'] += 1
                continue
            if result_type == 'folder':
                self.logger.info(f'Current folder in share "{share_name}": "{next_fullpath}"')
                self.spider_folder(share_name, next_fullpath + '/')
            else:
                self.logger.info(f'Current file in share "{share_name}": "{next_fullpath}"')
                self.parse_file(share_name, next_fullpath, result)

    def parse_file(self, share_name, file_path, file_info):
        if False:
            while True:
                i = 10
        'This function checks file attributes against various filters, records file metadata,\n        and downloads eligible files if the download flag is set.\n        '
        file_size = file_info.get_filesize()
        file_creation_time = file_info.get_ctime_epoch()
        file_modified_time = file_info.get_mtime_epoch()
        file_access_time = file_info.get_atime_epoch()
        self.results[share_name][file_path] = {'size': human_size(file_size), 'ctime_epoch': human_time(file_creation_time), 'mtime_epoch': human_time(file_modified_time), 'atime_epoch': human_time(file_access_time)}
        self.stats['file_sizes'].append(file_size)
        if not self.download_flag:
            return
        (_, file_extension) = os.path.splitext(file_path)
        if file_extension:
            self.stats['file_exts'].add(file_extension.lower())
            if file_extension.lower() in self.exclude_exts:
                self.logger.info(f'The file "{file_path}" has an excluded extension.')
                self.stats['num_files_filtered'] += 1
                return
        if file_size > self.max_file_size:
            self.logger.info(f'File {file_path} has size {human_size(file_size)} > max size {human_size(self.max_file_size)}.')
            self.stats['num_files_filtered'] += 1
            return
        remote_file = self.get_remote_file(share_name, file_path)
        if not remote_file:
            self.logger.fail(f'Cannot read remote file "{file_path}".')
            self.stats['num_get_fail'] += 1
            return
        (file_dir, file_name) = self.get_file_save_path(remote_file)
        download_path = os.path.join(file_dir, file_name)
        needs_update_flag = False
        if os.path.exists(download_path):
            if file_modified_time <= os.stat(download_path).st_mtime and os.path.getsize(download_path) == file_size:
                self.logger.info(f'File already downloaded "{file_path}" => "{download_path}".')
                self.stats['num_files_unmodified'] += 1
                return
            else:
                needs_update_flag = True
        download_success = False
        try:
            self.logger.info(f'Downloading file "{file_path}" => "{download_path}".')
            remote_file.open()
            self.save_file(remote_file, share_name)
            remote_file.close()
            download_success = True
        except SessionError as e:
            if 'STATUS_SHARING_VIOLATION' in str(e):
                pass
        except Exception as e:
            self.logger.fail(f'Failed to download file "{file_path}". Error: {str(e)}')
        if download_success:
            self.stats['num_get_success'] += 1
            if needs_update_flag:
                self.stats['num_files_updated'] += 1
        else:
            self.stats['num_get_fail'] += 1

    def save_file(self, remote_file, share_name):
        if False:
            return 10
        'This function reads the `remote_file` in chunks using the `read_chunk` method.\n        Each chunk is then written to the local file until the entire file is saved.\n        It handles cases where the file remains empty due to errors.\n        '
        remote_file.seek(0, 0)
        (folder, filename) = self.get_file_save_path(remote_file)
        download_path = os.path.join(folder, filename)
        self.logger.debug(f'Create folder "{folder}"')
        make_dirs(folder)
        try:
            with open(download_path, 'wb') as fd:
                while True:
                    chunk = self.read_chunk(remote_file)
                    if not chunk:
                        break
                    fd.write(chunk)
        except Exception as e:
            self.logger.fail(f'Error writing file "{remote_path}" from share "{share_name}": {e}')
        if os.path.getsize(download_path) == 0 and remote_file.get_filesize() > 0:
            os.remove(download_path)
            remote_path = str(remote_file)[2:]
            self.logger.fail(f'Unable to download file "{remote_path}".')

    def dump_folder_metadata(self, results):
        if False:
            print('Hello World!')
        'This function takes the metadata results as input and writes them to a JSON file\n        in the `self.output_folder`. The results are formatted with indentation and\n        sorted keys before being written to the file.\n        '
        metadata_path = os.path.join(self.output_folder, f'{self.host}.json')
        try:
            with open(metadata_path, 'w', encoding='utf-8') as fd:
                fd.write(json.dumps(results, indent=4, sort_keys=True))
            self.logger.success(f'Saved share-file metadata to "{metadata_path}".')
        except Exception as e:
            self.logger.fail(f'Failed to save share metadata: {str(e)}')

    def print_stats(self):
        if False:
            return 10
        'This function prints the statistics during processing.'
        shares = self.stats.get('shares', [])
        if shares:
            num_shares = len(shares)
            shares_str = ', '.join(shares)
            self.logger.display(f'SMB Shares:           {num_shares} ({shares_str})')
        shares_readable = self.stats.get('shares_readable', [])
        if shares_readable:
            num_readable_shares = len(shares_readable)
            if len(shares_readable) > 10:
                shares_readable_str = ', '.join(shares_readable[:10]) + '...'
            else:
                shares_readable_str = ', '.join(shares_readable)
            self.logger.display(f'SMB Readable Shares:  {num_readable_shares} ({shares_readable_str})')
        shares_writable = self.stats.get('shares_writable', [])
        if shares_writable:
            num_writable_shares = len(shares_writable)
            if len(shares_writable) > 10:
                shares_writable_str = ', '.join(shares_writable[:10]) + '...'
            else:
                shares_writable_str = ', '.join(shares_writable)
            self.logger.display(f'SMB Writable Shares:  {num_writable_shares} ({shares_writable_str})')
        num_shares_filtered = self.stats.get('num_shares_filtered', 0)
        if num_shares_filtered:
            self.logger.display(f'SMB Filtered Shares:  {num_shares_filtered}')
        num_folders = self.stats.get('num_folders', 0)
        self.logger.display(f'Total folders found:  {num_folders}')
        num_folders_filtered = self.stats.get('num_folders_filtered', 0)
        if num_folders_filtered:
            num_filtered_folders = len(num_folders_filtered)
            self.logger.display(f'Folders Filtered:     {num_filtered_folders}')
        num_files = self.stats.get('num_files', 0)
        self.logger.display(f'Total files found:    {num_files}')
        num_files_filtered = self.stats.get('num_files_filtered', 0)
        if num_files_filtered:
            self.logger.display(f'Files filtered:       {num_files_filtered}')
        if num_files == 0:
            return
        file_sizes = self.stats.get('file_sizes', [])
        if file_sizes:
            total_file_size = sum(file_sizes)
            min_file_size = min(file_sizes)
            max_file_size = max(file_sizes)
            average_file_size = total_file_size / num_files
            self.logger.display(f'File size average:    {human_size(average_file_size)}')
            self.logger.display(f'File size min:        {human_size(min_file_size)}')
            self.logger.display(f'File size max:        {human_size(max_file_size)}')
        file_exts = list(self.stats.get('file_exts', []))
        if file_exts:
            num_unique_file_exts = len(file_exts)
            if len(file_exts) > 10:
                unique_exts_str = ', '.join(file_exts[:10]) + '...'
            else:
                unique_exts_str = ', '.join(file_exts)
            self.logger.display(f'File unique exts:     {num_unique_file_exts} ({unique_exts_str})')
        if self.download_flag:
            num_get_success = self.stats.get('num_get_success', 0)
            if num_get_success:
                self.logger.display(f'Downloads successful: {num_get_success}')
            num_get_fail = self.stats.get('num_get_fail', 0)
            if num_get_fail:
                self.logger.display(f'Downloads failed:     {num_get_fail}')
            num_files_unmodified = self.stats.get('num_files_unmodified', 0)
            if num_files_unmodified:
                self.logger.display(f'Unmodified files:     {num_files_unmodified}')
            num_files_updated = self.stats.get('num_files_updated', 0)
            if num_files_updated:
                self.logger.display(f'Updated files:        {num_files_updated}')
            if num_files_unmodified and (not num_files_updated):
                self.logger.display('All files were not changed.')
            if num_files_filtered == num_files:
                self.logger.display('All files were ignored.')
            if num_get_fail == 0:
                self.logger.success('All files processed successfully.')

class CMEModule:
    """
    Spider plus module
    Module by @vincd
    Updated by @godylockz
    """
    name = 'spider_plus'
    description = 'List files recursively (excluding `EXCLUDE_FILTER` and `EXCLUDE_EXTS` extensions) and save JSON share-file metadata to the `OUTPUT_FOLDER`. If `DOWNLOAD_FLAG`=True, download files smaller then `MAX_FILE_SIZE` to the `OUTPUT_FOLDER`.'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            while True:
                i = 10
        '\n        DOWNLOAD_FLAG     Download all share folders/files (Default: False)\n        STATS_FLAG        Disable file/download statistics (Default: True)\n        EXCLUDE_EXTS      Case-insensitive extension filter to exclude (Default: ico,lnk)\n        EXCLUDE_FILTER    Case-insensitive filter to exclude folders/files (Default: print$,ipc$)\n        MAX_FILE_SIZE     Max file size to download (Default: 51200)\n        OUTPUT_FOLDER     Path of the local folder to save files (Default: /tmp/cme_spider_plus)\n        '
        self.download_flag = False
        if any(('DOWNLOAD' in key for key in module_options.keys())):
            self.download_flag = True
        self.stats_flag = True
        if any(('STATS' in key for key in module_options.keys())):
            self.stats_flag = False
        self.exclude_exts = get_list_from_option(module_options.get('EXCLUDE_EXTS', 'ico,lnk'))
        self.exclude_exts = [d.lower() for d in self.exclude_exts]
        self.exclude_filter = get_list_from_option(module_options.get('EXCLUDE_FILTER', 'print$,ipc$'))
        self.exclude_filter = [d.lower() for d in self.exclude_filter]
        self.max_file_size = int(module_options.get('MAX_FILE_SIZE', 50 * 1024))
        self.output_folder = module_options.get('OUTPUT_FOLDER', os.path.join('/tmp', 'cme_spider_plus'))

    def on_login(self, context, connection):
        if False:
            return 10
        context.log.display('Started module spidering_plus with the following options:')
        context.log.display(f' DOWNLOAD_FLAG: {self.download_flag}')
        context.log.display(f'    STATS_FLAG: {self.stats_flag}')
        context.log.display(f'EXCLUDE_FILTER: {self.exclude_filter}')
        context.log.display(f'  EXCLUDE_EXTS: {self.exclude_exts}')
        context.log.display(f' MAX_FILE_SIZE: {human_size(self.max_file_size)}')
        context.log.display(f' OUTPUT_FOLDER: {self.output_folder}')
        spider = SMBSpiderPlus(connection, context.log, self.download_flag, self.stats_flag, self.exclude_exts, self.exclude_filter, self.max_file_size, self.output_folder)
        spider.spider_shares()