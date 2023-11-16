"""
Loki
Simple IOC Scanner

Detection is based on three detection methods:

1. File Name IOC
   Applied to file names

2. Yara Check
   Applied to files and processes

3. Hash Check
   Compares known malicious hashes with the ones of the scanned files

Loki combines all IOCs from ReginScanner and SkeletonKeyScanner and is the
little brother of THOR our full-featured corporate APT Scanner

Florian Roth

DISCLAIMER - USE AT YOUR OWN RISK.
"""
import sys
import os
import argparse
import traceback
import yara
import re
import stat
import psutil
import platform
import signal as signal_module
from sys import platform as _platform
from subprocess import Popen
from collections import Counter
import datetime
from bisect import bisect_left
from lib.lokilogger import *
from lib.levenshtein import LevCheck
from lib.helpers import *
from lib.pesieve import PESieve
from lib.doublepulsar import DoublePulsar
from lib.vuln_checker import VulnChecker
os_platform = ''
if _platform == 'linux' or _platform == 'linux2':
    os_platform = 'linux'
elif _platform == 'darwin':
    os_platform = 'macos'
elif _platform == 'win32':
    os_platform = 'windows'
if os_platform == 'windows':
    try:
        import wmi
        import win32api
        from win32com.shell import shell
        import win32file
    except Exception:
        print('Linux System - deactivating process memory check ...')
        os_platform = 'linux'
if os_platform == '':
    print('Unable to determine platform - LOKI is lost.')
    sys.exit(1)
EVIL_EXTENSIONS = ['.vbs', '.ps', '.ps1', '.rar', '.tmp', '.bas', '.bat', '.chm', '.cmd', '.com', '.cpl', '.crt', '.dll', '.exe', '.hta', '.js', '.lnk', '.msc', '.ocx', '.pcd', '.pif', '.pot', '.pdf', '.reg', '.scr', '.sct', '.sys', '.url', '.vb', '.vbe', '.wsc', '.wsf', '.wsh', '.ct', '.t', '.input', '.war', '.jsp', '.jspx', '.php', '.asp', '.aspx', '.doc', '.docx', '.pdf', '.xls', '.xlsx', '.ppt', '.pptx', '.tmp', '.log', '.dump', '.pwd', '.w', '.txt', '.conf', '.cfg', '.conf', '.config', '.psd1', '.psm1', '.ps1xml', '.clixml', '.psc1', '.pssc', '.pl', '.www', '.rdp', '.jar', '.docm', '.sys']
SCRIPT_EXTENSIONS = ['.asp', '.vbs', '.ps1', '.bas', '.bat', '.js', '.vb', '.vbe', '.wsc', '.wsf', '.wsh', '.jsp', '.jspx', '.php', '.asp', '.aspx', '.psd1', '.psm1', '.ps1xml', '.clixml', '.psc1', '.pssc', '.pl']
SCRIPT_TYPES = ['VBS', 'PHP', 'JSP', 'ASP', 'BATCH']

def ioc_contains(sorted_list, value):
    if False:
        for i in range(10):
            print('nop')
    index = bisect_left(sorted_list, value)
    return index != len(sorted_list) and sorted_list[index] == value

class Loki(object):
    yara_rules = []
    filename_iocs = []
    hashes_md5 = {}
    hashes_sha1 = {}
    hashes_sha256 = {}
    hashes_scores = {}
    false_hashes = {}
    c2_server = {}
    yara_rule_directories = []
    fullExcludes = []
    startExcludes = []
    filetype_magics = {}
    max_filetype_magics = 0
    LINUX_PATH_SKIPS_START = set(['/proc', '/dev', '/sys/kernel/debug', '/sys/kernel/slab', '/sys/devices', '/usr/src/linux'])
    MOUNTED_DEVICES = set(['/media', '/volumes'])
    LINUX_PATH_SKIPS_END = set(['/initctl'])

    def __init__(self, intense_mode):
        if False:
            i = 10
            return i + 15
        self.intense_mode = intense_mode
        self.app_path = get_application_path()
        if os_platform == 'windows':
            self.peSieve = PESieve(self.app_path, is64bit(), logger)
        sig_dir = os.path.join(self.app_path, 'signature-base')
        if not os.path.exists(sig_dir) or os.listdir(sig_dir) == []:
            logger.log('NOTICE', 'Init', "The 'signature-base' subdirectory doesn't exist or is empty. Trying to retrieve the signature database automatically.")
            updateLoki(sigsOnly=True)
        self.initialize_excludes(os.path.join(self.app_path, 'config/excludes.cfg'.replace('/', os.sep)))
        if not args.force:
            if os_platform == 'linux' and args.alldrives:
                self.startExcludes = self.LINUX_PATH_SKIPS_START
            elif os_platform == 'linux':
                self.startExcludes = self.LINUX_PATH_SKIPS_START | self.MOUNTED_DEVICES | set(getExcludedMountpoints())
            if os_platform == 'macos' and args.alldrives:
                self.startExcludes = self.LINUX_PATH_SKIPS_START
            elif os_platform == 'macos':
                self.startExcludes = self.LINUX_PATH_SKIPS_START | self.MOUNTED_DEVICES
        self.ioc_path = os.path.join(self.app_path, 'signature-base/iocs/'.replace('/', os.sep))
        self.yara_rule_directories.append(os.path.join(self.app_path, 'signature-base/yara'.replace('/', os.sep)))
        self.yara_rule_directories.append(os.path.join(self.app_path, 'signature-base/iocs/yara'.replace('/', os.sep)))
        self.yara_rule_directories.append(os.path.join(self.app_path, 'signature-base/3rdparty'.replace('/', os.sep)))
        self.initialize_filename_iocs(self.ioc_path)
        logger.log('INFO', 'Init', 'File Name Characteristics initialized with %s regex patterns' % len(self.filename_iocs))
        self.initialize_c2_iocs(self.ioc_path)
        logger.log('INFO', 'Init', 'C2 server indicators initialized with %s elements' % len(self.c2_server.keys()))
        self.initialize_hash_iocs(self.ioc_path)
        logger.log('INFO', 'Init', 'Malicious MD5 Hashes initialized with %s hashes' % len(self.hashes_md5.keys()))
        logger.log('INFO', 'Init', 'Malicious SHA1 Hashes initialized with %s hashes' % len(self.hashes_sha1.keys()))
        logger.log('INFO', 'Init', 'Malicious SHA256 Hashes initialized with %s hashes' % len(self.hashes_sha256.keys()))
        self.initialize_hash_iocs(self.ioc_path, false_positive=True)
        logger.log('INFO', 'Init', 'False Positive Hashes initialized with %s hashes' % len(self.false_hashes.keys()))
        self.initialize_yara_rules()
        self.initialize_filetype_magics(os.path.join(self.app_path, 'signature-base/misc/file-type-signatures.txt'.replace('/', os.sep)))
        self.LevCheck = LevCheck()

    def scan_path(self, path):
        if False:
            i = 10
            return i + 15
        if not os.path.exists(path):
            logger.log('ERROR', 'FileScan', 'None Existing Scanning Path %s ...  ' % path)
            return
        logger.log('INFO', 'FileScan', 'Scanning Path %s ...  ' % path)
        for skip in self.startExcludes:
            if path.startswith(skip):
                logger.log('INFO', 'FileScan', 'Skipping %s directory [fixed excludes] (try using --force, --allhds or --alldrives)' % skip)
                return
        c = 0
        for (root, directories, files) in os.walk(path, onerror=walk_error, followlinks=False):
            newDirectories = []
            for dir in directories:
                skipIt = False
                completePath = os.path.join(root, dir).lower() + os.sep
                for skip in self.startExcludes:
                    if completePath.startswith(skip):
                        logger.log('INFO', 'FileScan', 'Skipping %s directory [fixed excludes] (try using --force, --allhds or --alldrives)' % skip)
                        skipIt = True
                if not skipIt:
                    newDirectories.append(dir)
            directories[:] = newDirectories
            for filename in files:
                try:
                    reasons = []
                    total_score = 0
                    filePath = os.path.join(root, filename)
                    fpath = os.path.split(filePath)[0]
                    filePathCleaned = fpath.encode('ascii', errors='replace')
                    fileNameCleaned = filename.encode('ascii', errors='replace')
                    extension = os.path.splitext(filePath)[1].lower()
                    skipIt = False
                    for skip in self.fullExcludes:
                        if skip.search(filePath):
                            logger.log('DEBUG', 'FileScan', 'Skipping element %s' % filePath)
                            skipIt = True
                    if os_platform == 'linux' or os_platform == 'macos':
                        for skip in self.LINUX_PATH_SKIPS_END:
                            if filePath.endswith(skip):
                                if self.LINUX_PATH_SKIPS_END[skip] == 0:
                                    logger.log('INFO', 'FileScan', 'Skipping %s element' % skip)
                                    self.LINUX_PATH_SKIPS_END[skip] = 1
                                    skipIt = True
                        mode = os.stat(filePath).st_mode
                        if stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode) or stat.S_ISLNK(mode) or stat.S_ISSOCK(mode):
                            continue
                    if skipIt:
                        continue
                    c += 1
                    if not args.noindicator:
                        printProgress(c)
                    if self.app_path.lower() in filePath.lower():
                        logger.log('DEBUG', 'FileScan', 'Skipping file in program directory FILE: %s' % filePathCleaned)
                        continue
                    fileSize = os.stat(filePath).st_size
                    for fioc in self.filename_iocs:
                        match = fioc['regex'].search(filePath)
                        if match:
                            if fioc['regex_fp']:
                                match_fp = fioc['regex_fp'].search(filePath)
                                if match_fp:
                                    continue
                            reasons.append('File Name IOC matched PATTERN: %s SUBSCORE: %s DESC: %s' % (fioc['regex'].pattern, fioc['score'], fioc['description']))
                            total_score += int(fioc['score'])
                    if not args.nolevcheck:
                        result = self.LevCheck.check(filename)
                        if result:
                            reasons.append('Levenshtein check - filename looks much like a well-known system file SUBSCORE: 40 ORIGINAL: %s' % result)
                            total_score += 60
                    firstBytes = b''
                    firstBytesString = b'-'
                    hashString = ''
                    try:
                        with open(filePath, 'rb') as f:
                            firstBytes = f.read(4)
                    except Exception:
                        logger.log('DEBUG', 'FileScan', 'Cannot open file %s (access denied)' % filePathCleaned)
                    fileType = get_file_type(filePath, self.filetype_magics, self.max_filetype_magics, logger)
                    do_intense_check = True
                    if not self.intense_mode and fileType == 'UNKNOWN' and (extension not in EVIL_EXTENSIONS):
                        if args.printall:
                            logger.log('INFO', 'FileScan', 'Skipping file due to fast scan mode: %s' % fileNameCleaned)
                        do_intense_check = False
                    fileData = ''
                    print_filesize_info = False
                    fileSizeLimit = int(args.s) * 1024
                    if fileSize > fileSizeLimit:
                        do_intense_check = False
                        print_filesize_info = True
                    if fileType == 'MDMP':
                        do_intense_check = True
                        print_filesize_info = False
                    if do_intense_check:
                        if args.printall:
                            logger.log('INFO', 'FileScan', 'Scanning %s TYPE: %s SIZE: %s' % (fileNameCleaned, fileType, fileSize))
                    elif args.printall:
                        logger.log('INFO', 'FileScan', 'Checking %s TYPE: %s SIZE: %s' % (fileNameCleaned, fileType, fileSize))
                    if print_filesize_info and args.printall:
                        logger.log('INFO', 'FileScan', 'Skipping file due to file size: %s TYPE: %s SIZE: %s CURRENT SIZE LIMIT(kilobytes): %d' % (fileNameCleaned, fileType, fileSize, fileSizeLimit))
                    if do_intense_check:
                        fileData = self.get_file_data(filePath)
                        firstBytesString = '%s / %s' % (fileData[:20].hex(), removeNonAsciiDrop(fileData[:20]))
                        matchType = None
                        matchDesc = None
                        matchHash = None
                        md5 = 0
                        sha1 = 0
                        sha256 = 0
                        (md5, sha1, sha256) = generateHashes(fileData)
                        md5_num = int(md5, 16)
                        sha1_num = int(sha1, 16)
                        sha256_num = int(sha256, 16)
                        if md5_num in self.false_hashes.keys() or sha1_num in self.false_hashes.keys() or sha256_num in self.false_hashes.keys():
                            continue
                        matchScore = 100
                        matchLevel = 'Malware'
                        if ioc_contains(self.hashes_md5_list, md5_num):
                            matchType = 'MD5'
                            matchDesc = self.hashes_md5[md5_num]
                            matchHash = md5
                            matchScore = self.hashes_scores[md5_num]
                        if ioc_contains(self.hashes_sha1_list, sha1_num):
                            matchType = 'SHA1'
                            matchDesc = self.hashes_sha1[sha1_num]
                            matchHash = sha1
                            matchScore = self.hashes_scores[sha1_num]
                        if ioc_contains(self.hashes_sha256_list, sha256_num):
                            matchType = 'SHA256'
                            matchDesc = self.hashes_sha256[sha256_num]
                            matchHash = sha256
                            matchScore = self.hashes_scores[sha256_num]
                        if matchScore < 80:
                            matchLevel = 'Suspicious'
                        hashString = 'MD5: %s SHA1: %s SHA256: %s' % (md5, sha1, sha256)
                        if matchType:
                            reasons.append('%s Hash TYPE: %s HASH: %s SUBSCORE: %d DESC: %s' % (matchLevel, matchType, matchHash, matchScore, matchDesc))
                            total_score += matchScore
                        if args.scriptanalysis:
                            if extension in SCRIPT_EXTENSIONS or type in SCRIPT_TYPES:
                                logger.log('DEBUG', 'FileScan', 'Performing character analysis on file %s ... ' % filePath)
                                (message, score) = self.script_stats_analysis(fileData)
                                if message:
                                    reasons.append('%s SCORE: %s' % (message, score))
                                    total_score += score
                        if fileType == 'MDMP':
                            logger.log('INFO', 'FileScan', 'Scanning memory dump file %s' % fileNameCleaned.decode('utf-8'))
                        try:
                            for (score, rule, description, reference, matched_strings, author) in self.scan_data(fileData=fileData, fileType=fileType, fileName=fileNameCleaned, filePath=filePathCleaned, extension=extension, md5=md5):
                                message = 'Yara Rule MATCH: %s SUBSCORE: %s DESCRIPTION: %s REF: %s AUTHOR: %s' % (rule, score, description, reference, author)
                                if len(matched_strings) > 0:
                                    message += ' MATCHES: %s' % ', '.join(matched_strings)
                                total_score += score
                                reasons.append(message)
                        except Exception:
                            if logger.debug:
                                traceback.print_exc()
                            logger.log('ERROR', 'FileScan', 'Cannot YARA scan file: %s' % filePathCleaned)
                    fileInfo = 'FILE: %s SCORE: %s TYPE: %s SIZE: %s FIRST_BYTES: %s %s %s ' % (filePath, total_score, fileType, fileSize, firstBytesString, hashString, getAgeString(filePath))
                    if total_score >= args.a:
                        message_type = 'ALERT'
                    elif total_score >= args.w:
                        message_type = 'WARNING'
                    elif total_score >= args.n:
                        message_type = 'NOTICE'
                    if total_score < args.n:
                        continue
                    message_body = fileInfo
                    for (i, r) in enumerate(reasons):
                        if i < 2 or args.allreasons:
                            message_body += 'REASON_{0}: {1}'.format(i + 1, r)
                    logger.log(message_type, 'FileScan', message_body)
                except Exception:
                    if logger.debug:
                        traceback.print_exc()
                        sys.exit(1)

    def scan_data(self, fileData, fileType='-', fileName=b'-', filePath=b'-', extension=b'-', md5='-'):
        if False:
            return 10
        try:
            for rules in self.yara_rules:
                matches = rules.match(data=fileData, externals={'filename': fileName.decode('utf-8'), 'filepath': filePath.decode('utf-8'), 'extension': extension, 'filetype': fileType, 'md5': md5, 'owner': 'dummy'})
                if matches:
                    for match in matches:
                        score = 70
                        description = 'not set'
                        reference = '-'
                        author = '-'
                        if hasattr(match, 'meta'):
                            if 'description' in match.meta:
                                description = match.meta['description']
                            if 'cluster' in match.meta:
                                description = 'IceWater Cluster {0}'.format(match.meta['cluster'])
                            if 'reference' in match.meta:
                                reference = match.meta['reference']
                            if 'viz_url' in match.meta:
                                reference = match.meta['viz_url']
                            if 'author' in match.meta:
                                author = match.meta['author']
                            if 'score' in match.meta:
                                score = int(match.meta['score'])
                        matched_strings = []
                        if hasattr(match, 'strings'):
                            matched_strings = self.get_string_matches(match.strings)
                        yield (score, match.rule, description, reference, matched_strings, author)
        except Exception:
            if logger.debug:
                traceback.print_exc()

    def get_string_matches(self, strings):
        if False:
            while True:
                i = 10
        try:
            matching_strings = []
            for string in strings:
                string_value = str(string.instances[0]).replace("'", '\\')
                if len(string_value) > 140:
                    string_value = string_value[:140] + ' ... (truncated)'
                matching_strings.append("{0}: '{1}'".format(string.identifier, string_value))
            return matching_strings
        except:
            traceback.print_exc()

    def check_svchost_owner(self, owner):
        if False:
            while True:
                i = 10
        import ctypes
        import locale
        windll = ctypes.windll.kernel32
        locale = locale.windows_locale[windll.GetUserDefaultUILanguage()]
        if locale == 'fr_FR':
            return owner.upper().startswith('SERVICE LOCAL') or owner.upper().startswith(u'SERVICE RÉSEAU') or re.match('SERVICE R.SEAU', owner) or (owner == u'Système') or owner.upper().startswith(u'AUTORITE NT\\Système') or re.match('AUTORITE NT\\\\Syst.me', owner)
        elif locale == 'ru_RU':
            return owner.upper().startswith('NET') or owner == u'система' or owner.upper().startswith('LO')
        else:
            return owner.upper().startswith('NT ') or owner.upper().startswith('NET') or owner.upper().startswith('LO') or owner.upper().startswith('SYSTEM')

    def scan_processes(self, nopesieve, nolisten, excludeprocess, pesieveshellc):
        if False:
            for i in range(10):
                print('nop')
        c = wmi.WMI()
        processes = c.Win32_Process()
        t_systemroot = os.environ['SYSTEMROOT']
        wininit_pid = 0
        lsass_count = 0
        loki_pid = os.getpid()
        loki_ppid = psutil.Process(os.getpid()).ppid()
        for process in processes:
            try:
                if process.name.lower() in excludeprocess:
                    continue
                pid = process.ProcessId
                name = process.Name
                cmd = process.CommandLine
                if not cmd:
                    cmd = 'N/A'
                if not name:
                    name = 'N/A'
                path = 'none'
                parent_pid = process.ParentProcessId
                priority = process.Priority
                ws_size = process.VirtualSize
                if process.ExecutablePath:
                    path = process.ExecutablePath
                try:
                    owner_raw = process.GetOwner()
                    owner = owner_raw[2]
                except Exception:
                    owner = 'unknown'
                if not owner:
                    owner = 'unknown'
            except Exception:
                logger.log('ALERT', 'ProcessScan', "Error getting all process information. Did you run the scanner 'As Administrator'?")
                continue
            if name == 'wininit.exe':
                wininit_pid = pid
            if '\\' not in cmd and path != 'none':
                cmd = path
            process_info = 'PID: %s NAME: %s OWNER: %s CMD: %s PATH: %s' % (str(pid), name, owner, cmd, path)
            if pid == 0 or pid == 4:
                logger.log('INFO', 'ProcessScan', 'Skipping Process %s' % process_info)
                continue
            if loki_pid == pid or loki_ppid == pid:
                logger.log('INFO', 'ProcessScan', 'Skipping LOKI Process %s' % process_info)
                continue
            logger.log('INFO', 'ProcessScan', 'Scanning Process %s' % process_info)
            if re.search('psexec .* [a-fA-F0-9]{32}', cmd, re.IGNORECASE):
                logger.log('WARNING', 'ProcessScan', 'Process that looks liks SKELETON KEY psexec execution detected %s' % process_info)
            for fioc in self.filename_iocs:
                match = fioc['regex'].search(cmd)
                if match:
                    if int(fioc['score']) > 70:
                        logger.log('ALERT', 'ProcessScan', 'File Name IOC matched PATTERN: %s DESC: %s MATCH: %s' % (fioc['regex'].pattern, fioc['description'], cmd))
                    elif int(fioc['score']) > 40:
                        logger.log('WARNING', 'ProcessScan', 'File Name Suspicious IOC matched PATTERN: %s DESC: %s MATCH: %s' % (fioc['regex'].pattern, fioc['description'], cmd))
            if name == 'waitfor.exe':
                logger.log('WARNING', 'ProcessScan', 'Suspicious waitfor.exe process https://twitter.com/subTee/status/872274262769500160 %s' % process_info)
            if processExists(pid):
                if int(ws_size) < args.maxworkingset * 1048576:
                    try:
                        alerts = []
                        for rules in self.yara_rules:
                            matches = rules.match(pid=pid)
                            if matches:
                                for match in matches:
                                    memory_rule = 1
                                    if hasattr(match, 'meta'):
                                        if 'memory' in match.meta:
                                            memory_rule = int(match.meta['memory'])
                                    if memory_rule == 1:
                                        alerts.append('Yara Rule MATCH: %s %s' % (match.rule, process_info))
                        if len(alerts) > 5:
                            logger.log('WARNING', 'ProcessScan', 'Too many matches on process memory - most likely a false positive %s' % process_info)
                        elif len(alerts) > 0:
                            for alert in alerts:
                                logger.log('ALERT', 'ProcessScan', alert)
                    except Exception:
                        if logger.debug:
                            traceback.print_exc()
                        if path != 'none':
                            logger.log('ERROR', 'ProcessScan', "Error during process memory Yara check (maybe the process doesn't exist anymore or access denied) %s" % process_info)
                else:
                    logger.log('DEBUG', 'ProcessScan', "Skipped Yara memory check due to the process' big working set size (stability issues) PID: %s NAME: %s SIZE: %s" % (pid, name, ws_size))
            try:
                if processExists(pid) and self.peSieve.active and (not nopesieve):
                    logger.log('DEBUG', 'ProcessScan', 'PE-Sieve scan of process PID: %s' % pid)
                    results = self.peSieve.scan(pid, pesieveshellc)
                    if results['replaced']:
                        logger.log('WARNING', 'ProcessScan', 'PE-Sieve reported replaced process %s REPLACED: %s' % (process_info, str(results['replaced'])))
                    elif results['implanted_pe'] or results['implanted_shc']:
                        logger.log('WARNING', 'ProcessScan', 'PE-Sieve reported implanted process %s IMPLANTED PE: %s IMPLANTED SHC: %s' % (process_info, str(results['implanted_pe']), str(results['implanted_shc'])))
                    elif results['patched']:
                        logger.log('NOTICE', 'ProcessScan', 'PE-Sieve reported patched process %s PATCHED: %s' % (process_info, str(results['patched'])))
                    elif results['unreachable_file']:
                        logger.log('NOTICE', 'ProcessScan', 'PE-Sieve reported a process with unreachable exe %s UNREACHABLE: %s' % (process_info, str(results['unreachable_file'])))
                    else:
                        logger.log('INFO', 'ProcessScan', 'PE-Sieve reported no anomalies %s' % process_info)
            except WindowsError:
                if logger.debug:
                    traceback.print_exc()
                logger.log('ERROR', 'ProcessScan', 'Error while accessing process handle using PE-Sieve (use --debug for full traceback)')
            if not nolisten:
                self.check_process_connections(process)
            if name == 'System' and (not pid == 4):
                logger.log('WARNING', 'ProcessScan', 'System process without PID=4 %s' % process_info)
            if name == 'smss.exe' and (not parent_pid == 4):
                logger.log('WARNING', 'ProcessScan', 'smss.exe parent PID is != 4 %s' % process_info)
            if path != 'none':
                if name == 'smss.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'smss.exe path is not System32 %s' % process_info)
            if name == 'smss.exe' and priority != 11:
                logger.log('WARNING', 'ProcessScan', 'smss.exe priority is not 11 %s' % process_info)
            if path != 'none':
                if name == 'csrss.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'csrss.exe path is not System32 %s' % process_info)
            if name == 'csrss.exe' and priority != 13:
                logger.log('WARNING', 'ProcessScan', 'csrss.exe priority is not 13 %s' % process_info)
            if path != 'none':
                if name == 'wininit.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'wininit.exe path is not System32 %s' % process_info)
            if name == 'wininit.exe' and priority != 13:
                logger.log('NOTICE', 'ProcessScan', 'wininit.exe priority is not 13 %s' % process_info)
            if name == 'wininit.exe':
                wininit_pid = pid
            if path != 'none':
                if name == 'services.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'services.exe path is not System32 %s' % process_info)
            if name == 'services.exe' and priority != 9:
                logger.log('WARNING', 'ProcessScan', 'services.exe priority is not 9 %s' % process_info)
            if wininit_pid > 0:
                if name == 'services.exe' and (not parent_pid == wininit_pid):
                    logger.log('WARNING', 'ProcessScan', 'services.exe parent PID is not the one of wininit.exe %s' % process_info)
            if path != 'none':
                if name == 'lsass.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'lsass.exe path is not System32 %s' % process_info)
            if name == 'lsass.exe' and priority != 9:
                logger.log('WARNING', 'ProcessScan', 'lsass.exe priority is not 9 %s' % process_info)
            if wininit_pid > 0:
                if name == 'lsass.exe' and (not parent_pid == wininit_pid):
                    logger.log('WARNING', 'ProcessScan', 'lsass.exe parent PID is not the one of wininit.exe %s' % process_info)
            if name == 'lsass.exe':
                lsass_count += 1
                if lsass_count > 1:
                    logger.log('WARNING', 'ProcessScan', 'lsass.exe count is higher than 1 %s' % process_info)
            if path != 'none':
                if name == 'svchost.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'svchost.exe path is not System32 %s' % process_info)
            if name == 'svchost.exe' and priority != 8:
                logger.log('NOTICE', 'ProcessScan', 'svchost.exe priority is not 8 %s' % process_info)
            if name == 'svchost.exe' and ' -k ' not in cmd and (cmd != 'N/A'):
                logger.log('WARNING', 'ProcessScan', 'svchost.exe process does not contain a -k in its command line %s' % process_info)
            if path != 'none':
                if name == 'lsm.exe' and (not ('system32' in path.lower() or 'system32' in cmd.lower())):
                    logger.log('WARNING', 'ProcessScan', 'lsm.exe path is not System32 %s' % process_info)
            if name == 'lsm.exe' and priority != 8:
                logger.log('NOTICE', 'ProcessScan', 'lsm.exe priority is not 8 %s' % process_info)
            if name == 'lsm.exe' and (not (owner.startswith('NT ') or owner.startswith('LO') or owner.startswith('SYSTEM') or owner.startswith(u'система'))):
                logger.log(u'WARNING', 'ProcessScan', 'lsm.exe process owner is suspicious %s' % process_info)
            if wininit_pid > 0:
                if name == 'lsm.exe' and (not parent_pid == wininit_pid):
                    logger.log('WARNING', 'ProcessScan', 'lsm.exe parent PID is not the one of wininit.exe %s' % process_info)
            if name == 'winlogon.exe' and priority != 13:
                logger.log('WARNING', 'ProcessScan', 'winlogon.exe priority is not 13 %s' % process_info)
            if re.search('(Windows 7|Windows Vista)', getPlatformFull()):
                if name == 'winlogon.exe' and parent_pid > 0:
                    for proc in processes:
                        if parent_pid == proc.ProcessId:
                            logger.log('WARNING', 'ProcessScan', 'winlogon.exe has a parent ID but should have none %s PARENTID: %s' % (process_info, str(parent_pid)))
            if path != 'none':
                if name == 'explorer.exe' and t_systemroot.lower() not in path.lower():
                    logger.log('WARNING', 'ProcessScan', 'explorer.exe path is not %%SYSTEMROOT%% %s' % process_info)
            if name == 'explorer.exe' and parent_pid > 0:
                for proc in processes:
                    if parent_pid == proc.ProcessId:
                        logger.log('NOTICE', 'ProcessScan', 'explorer.exe has a parent ID but should have none %s' % process_info)

    def check_process_connections(self, process):
        if False:
            return 10
        try:
            MAXIMUM_CONNECTIONS = 20
            connection_count = 0
            pid = process.ProcessId
            name = process.Name
            try:
                p = psutil.Process(pid)
            except Exception:
                if logger.debug:
                    traceback.print_exc()
                return
            for x in p.connections():
                try:
                    command = process.CommandLine
                except Exception:
                    command = p.cmdline()
                if x.status == 'LISTEN':
                    connection_count += 1
                    logger.log('NOTICE', 'ProcessScan', 'Listening process PID: %s NAME: %s COMMAND: %s IP: %s PORT: %s' % (str(pid), name, command, str(x.laddr[0]), str(x.laddr[1])))
                    if str(x.laddr[1]) == '0':
                        logger.log('WARNING', 'ProcessScan', 'Listening on Port 0 PID: %s NAME: %s COMMAND: %s  IP: %s PORT: %s' % (str(pid), name, command, str(x.laddr[0]), str(x.laddr[1])))
                if x.status == 'ESTABLISHED':
                    (is_match, description) = self.check_c2(str(x.raddr[0]))
                    if is_match:
                        logger.log('ALERT', 'ProcessScan', 'Malware Domain/IP match in remote address PID: %s NAME: %s COMMAND: %s IP: %s PORT: %s DESC: %s' % (str(pid), name, command, str(x.raddr[0]), str(x.raddr[1]), description))
                    connection_count += 1
                    logger.log('NOTICE', 'ProcessScan', 'Established connection PID: %s NAME: %s COMMAND: %s LIP: %s LPORT: %s RIP: %s RPORT: %s' % (str(pid), name, command, str(x.laddr[0]), str(x.laddr[1]), str(x.raddr[0]), str(x.raddr[1])))
                if connection_count > MAXIMUM_CONNECTIONS:
                    logger.log('NOTICE', 'ProcessScan', 'Connection output threshold reached. Output truncated.')
                    return
        except Exception:
            if args.debug:
                traceback.print_exc()
                sys.exit(1)
            logger.log('INFO', 'ProcessScan', 'Process %s does not exist anymore or cannot be accessed' % str(pid))

    def check_rootkit(self):
        if False:
            i = 10
            return i + 15
        logger.log('INFO', 'Rootkit', 'Checking for Backdoors ...')
        dp = DoublePulsar(ip='127.0.0.1', timeout=None, verbose=args.debug)
        logger.log('INFO', 'Rootkit', 'Checking for Double Pulsar RDP Backdoor')
        try:
            (dp_rdp_result, message) = dp.check_ip_rdp()
            if dp_rdp_result:
                logger.log('ALERT', message)
            else:
                logger.log('INFO', 'Rootkit', 'Double Pulsar RDP check RESULT: %s' % message)
        except Exception:
            logger.log('INFO', 'Rootkit', 'Double Pulsar RDP check failed RESULT: Connection failure')
            if args.debug:
                traceback.print_exc()
        logger.log('INFO', 'Rootkit', 'Checking for Double Pulsar SMB Backdoor')
        try:
            (dp_smb_result, message) = dp.check_ip_smb()
            if dp_smb_result:
                logger.log('ALERT', message)
            else:
                logger.log('INFO', 'Rootkit', 'Double Pulsar SMB check RESULT: %s' % message)
        except Exception:
            logger.log('INFO', 'Rootkit', 'Double Pulsar SMB check failed RESULT: Connection failure')
            if args.debug:
                traceback.print_exc()

    def check_c2(self, remote_system):
        if False:
            for i in range(10):
                print('nop')
        if is_ip(remote_system):
            for c2 in self.c2_server:
                if is_cidr(c2):
                    if ip_in_net(remote_system, c2):
                        return (True, self.c2_server[c2])
                if c2 == remote_system:
                    return (True, self.c2_server[c2])
        else:
            for c2 in self.c2_server:
                if c2 in remote_system:
                    return (True, self.c2_server[c2])
        return (False, '')

    def initialize_c2_iocs(self, ioc_directory):
        if False:
            while True:
                i = 10
        try:
            for ioc_filename in os.listdir(ioc_directory):
                try:
                    if 'c2' in ioc_filename:
                        with codecs.open(os.path.join(ioc_directory, ioc_filename), 'r', encoding='utf-8') as file:
                            lines = file.readlines()
                            last_comment = ''
                            for line in lines:
                                try:
                                    if re.search('^#', line) or re.search('^[\\s]*$', line):
                                        last_comment = line.lstrip('#').lstrip(' ').rstrip('\n')
                                        continue
                                    if ';' in line:
                                        line = line.rstrip(' ').rstrip('\n\r')
                                        row = line.split(';')
                                        c2 = row[0]
                                    else:
                                        c2 = line
                                    if len(c2) < 4:
                                        logger.log('NOTICE', 'Init', 'C2 server definition is suspiciously short - will not add %s' % c2)
                                        continue
                                    self.c2_server[c2.lower()] = last_comment
                                except Exception:
                                    logger.log('ERROR', 'Init', 'Cannot read line: %s' % line)
                                    if logger.debug:
                                        sys.exit(1)
                except OSError:
                    logger.log('ERROR', 'Init', 'No such file or directory')
        except Exception:
            traceback.print_exc()
            logger.log('ERROR', 'Init', 'Error reading Hash file: %s' % ioc_filename)

    def initialize_filename_iocs(self, ioc_directory):
        if False:
            for i in range(10):
                print('nop')
        try:
            for ioc_filename in os.listdir(ioc_directory):
                if 'filename' in ioc_filename:
                    with codecs.open(os.path.join(ioc_directory, ioc_filename), 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        last_comment = ''
                        score = 0
                        desc = ''
                        for line in lines:
                            try:
                                if re.search('^[\\s]*$', line):
                                    continue
                                if re.search('^#', line):
                                    last_comment = line.lstrip('#').lstrip(' ').rstrip('\n')
                                    continue
                                if ';' in line:
                                    line = line.rstrip(' ').rstrip('\n\r')
                                    row = line.split(';')
                                    regex = row[0]
                                    score = row[1]
                                    if len(row) > 2:
                                        regex_fp = row[2]
                                    desc = last_comment
                                else:
                                    regex = line
                                regex = replaceEnvVars(regex)
                                regex = transformOS(regex, os_platform)
                                regex_fp_comp = None
                                if 'regex_fp' in locals():
                                    regex_fp = replaceEnvVars(regex_fp)
                                    regex_fp = transformOS(regex_fp, os_platform)
                                    regex_fp_comp = re.compile(regex_fp)
                                fioc = {'regex': re.compile(regex), 'score': score, 'description': desc, 'regex_fp': regex_fp_comp}
                                self.filename_iocs.append(fioc)
                            except Exception:
                                logger.log('ERROR', 'Init', 'Error reading line: %s' % line)
                                if logger.debug:
                                    traceback.print_exc()
                                    sys.exit(1)
        except Exception:
            if 'ioc_filename' in locals():
                logger.log('ERROR', 'Init', 'Error reading IOC file: %s' % ioc_filename)
            else:
                logger.log('ERROR', 'Init', 'Error reading files from IOC folder: %s' % ioc_directory)
                logger.log('ERROR', 'Init', 'Please make sure that you cloned the repo or downloaded the sub repository: See https://github.com/Neo23x0/Loki/issues/51')
            sys.exit(1)

    def initialize_yara_rules(self):
        if False:
            for i in range(10):
                print('nop')
        yaraRules = ''
        dummy = ''
        rule_count = 0
        try:
            for yara_rule_directory in self.yara_rule_directories:
                if not os.path.exists(yara_rule_directory):
                    continue
                logger.log('INFO', 'Init', 'Processing YARA rules folder {0}'.format(yara_rule_directory))
                for (root, directories, files) in os.walk(yara_rule_directory, onerror=walk_error, followlinks=False):
                    for file in files:
                        try:
                            yaraRuleFile = os.path.join(root, file)
                            if file.startswith('.') or file.startswith('~') or file.startswith('_'):
                                continue
                            extension = os.path.splitext(file)[1].lower()
                            if extension != '.yar' and extension != '.yara':
                                continue
                            with open(yaraRuleFile, 'r') as yfile:
                                yara_rule_data = yfile.read()
                            try:
                                compiledRules = yara.compile(source=yara_rule_data, externals={'filename': dummy, 'filepath': dummy, 'extension': dummy, 'filetype': dummy, 'md5': dummy, 'owner': dummy})
                                logger.log('DEBUG', 'Init', 'Initializing Yara rule %s' % file)
                                rule_count += 1
                            except Exception:
                                logger.log('ERROR', 'Init', 'Error while initializing Yara rule %s ERROR: %s' % (file, sys.exc_info()[1]))
                                traceback.print_exc()
                                if logger.debug:
                                    sys.exit(1)
                                continue
                            yaraRules += yara_rule_data
                        except Exception:
                            logger.log('ERROR', 'Init', 'Error reading signature file %s ERROR: %s' % (yaraRuleFile, sys.exc_info()[1]))
                            if logger.debug:
                                traceback.print_exc()
            try:
                logger.log('INFO', 'Init', 'Initializing all YARA rules at once (composed string of all rule files)')
                compiledRules = yara.compile(source=yaraRules, externals={'filename': dummy, 'filepath': dummy, 'extension': dummy, 'filetype': dummy, 'md5': dummy, 'owner': dummy})
                logger.log('INFO', 'Init', 'Initialized %d Yara rules' % rule_count)
            except Exception:
                traceback.print_exc()
                logger.log('ERROR', 'Init', 'Error during YARA rule compilation ERROR: %s - please fix the issue in the rule set' % sys.exc_info()[1])
                sys.exit(1)
            self.yara_rules.append(compiledRules)
        except Exception:
            logger.log('ERROR', 'Init', 'Error reading signature folder /signatures/')
            if logger.debug:
                traceback.print_exc()
                sys.exit(1)

    def initialize_hash_iocs(self, ioc_directory, false_positive=False):
        if False:
            for i in range(10):
                print('nop')
        HASH_WHITELIST = [int('d41d8cd98f00b204e9800998ecf8427e', 16), int('da39a3ee5e6b4b0d3255bfef95601890afd80709', 16), int('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 16), int('68b329da9893e34099c7d8ad5cb9c940', 16), int('adc83b19e793491b1c6ea0fd8b46cd9f32e592fc', 16), int('01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b', 16), int('81051bcc2cf1bedf378224b0a93e2877', 16), int('ba8ab5a0280b953aa97435ff8946cbcbb2755a27', 16), int('7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6', 16)]
        try:
            for ioc_filename in os.listdir(ioc_directory):
                if 'hash' in ioc_filename:
                    if false_positive and 'falsepositive' not in ioc_filename:
                        continue
                    with codecs.open(os.path.join(ioc_directory, ioc_filename), 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            try:
                                if re.search('^#', line) or re.search('^[\\s]*$', line):
                                    continue
                                row = line.split(';')
                                if len(row) == 3 and row[1].isdigit():
                                    hash = row[0].lower()
                                    score = int(row[1])
                                    comment = row[2].rstrip(' ').rstrip('\n')
                                else:
                                    hash = row[0].lower()
                                    comment = row[1].rstrip(' ').rstrip('\n')
                                    score = 100
                                if hash in HASH_WHITELIST:
                                    continue
                                self.hashes_scores[int(hash, 16)] = score
                                if len(hash) == 32:
                                    self.hashes_md5[int(hash, 16)] = comment
                                if len(hash) == 40:
                                    self.hashes_sha1[int(hash, 16)] = comment
                                if len(hash) == 64:
                                    self.hashes_sha256[int(hash, 16)] = comment
                                if false_positive:
                                    self.false_hashes[int(hash, 16)] = comment
                            except Exception:
                                if logger.debug:
                                    traceback.print_exc()
                                logger.log('ERROR', 'Init', 'Cannot read line: %s' % line)
                    if logger.debug:
                        logger.log('DEBUG', 'Init', 'Initialized %s hash IOCs from file %s' % (str(len(self.hashes_md5) + len(self.hashes_sha1) + len(self.hashes_sha256)), ioc_filename))
            self.hashes_md5_list = list(self.hashes_md5.keys())
            self.hashes_md5_list.sort()
            self.hashes_sha1_list = list(self.hashes_sha1.keys())
            self.hashes_sha1_list.sort()
            self.hashes_sha256_list = list(self.hashes_sha256.keys())
            self.hashes_sha256_list.sort()
        except Exception:
            if logger.debug:
                traceback.print_exc()
                sys.exit(1)
            logger.log('ERROR', 'Init', 'Error reading Hash file: %s' % ioc_filename)

    def initialize_filetype_magics(self, filetype_magics_file):
        if False:
            for i in range(10):
                print('nop')
        try:
            with open(filetype_magics_file, 'r') as config:
                lines = config.readlines()
            for line in lines:
                try:
                    if re.search('^#', line) or re.search('^[\\s]*$', line) or ';' not in line:
                        continue
                    (sig_raw, description) = line.rstrip('\n').split(';')
                    sig = re.sub(' ', '', sig_raw)
                    if len(sig) > self.max_filetype_magics:
                        self.max_filetype_magics = len(sig)
                    self.filetype_magics[sig] = description
                except Exception:
                    logger.log('ERROR', 'Init', 'Cannot read line: %s' % line)
        except Exception:
            if logger.debug:
                traceback.print_exc()
                sys.exit(1)
            logger.log('ERROR', 'Init', 'Error reading Hash file: %s' % filetype_magics_file)

    def initialize_excludes(self, excludes_file):
        if False:
            i = 10
            return i + 15
        try:
            excludes = []
            with open(excludes_file, 'r') as config:
                lines = config.read().splitlines()
            for line in lines:
                if re.search('^[\\s]*#', line):
                    continue
                try:
                    if re.search('\\w', line):
                        regex = re.compile(line, re.IGNORECASE)
                        excludes.append(regex)
                except Exception:
                    logger.log('ERROR', 'Init', 'Cannot compile regex: %s' % line)
            self.fullExcludes = excludes
        except Exception:
            if logger.debug:
                traceback.print_exc()
            logger.log('NOTICE', 'Init', 'Error reading excludes file: %s' % excludes_file)

    def get_file_data(self, filePath):
        if False:
            while True:
                i = 10
        fileData = b''
        try:
            with open(filePath, 'rb') as f:
                fileData = f.read()
        except Exception:
            if logger.debug:
                traceback.print_exc()
            logger.log('DEBUG', 'FileScan', 'Cannot open file %s (access denied)' % filePath)
        finally:
            return fileData

    def script_stats_analysis(self, data):
        if False:
            return 10
        '\n        Doing a statistical analysis for scripts like PHP, JavaScript or PowerShell to\n        detect obfuscated code\n        :param data:\n        :return: message, score\n        '
        anomal_chars = ['^', '{', '}', '"', ',', '<', '>', ';']
        anomal_char_stats = {}
        char_stats = {'upper': 0, 'lower': 0, 'numbers': 0, 'symbols': 0, 'spaces': 0}
        anomalies = []
        c = Counter(data)
        anomaly_score = 0
        for char in c.most_common():
            if char[0] in anomal_chars:
                anomal_char_stats[char[0]] = char[1]
            if char[0].isupper():
                char_stats['upper'] += char[1]
            elif char[0].islower():
                char_stats['lower'] += char[1]
            elif char[0].isdigit():
                char_stats['numbers'] += char[1]
            elif char[0].isspace():
                char_stats['spaces'] += char[1]
            else:
                char_stats['symbols'] += char[1]
        char_stats['total'] = len(data)
        char_stats['alpha'] = char_stats['upper'] + char_stats['lower']
        if char_stats['alpha'] > 40 and char_stats['upper'] > char_stats['lower'] * 0.9:
            anomalies.append('upper to lower ratio')
            anomaly_score += 20
        if char_stats['symbols'] > char_stats['alpha']:
            anomalies.append('more symbols than alphanum chars')
            anomaly_score += 40
        for (ac, count) in anomal_char_stats.iteritems():
            if count / char_stats['alpha'] > 0.05:
                anomalies.append("symbol count of '%s' very high" % ac)
                anomaly_score += 40
        message = "Anomaly detected ANOMALIES: '{0}'".format("', '".join(anomalies))
        if anomaly_score > 40:
            return (message, anomaly_score)
        return ('', 0)

def get_application_path():
    if False:
        for i in range(10):
            print('nop')
    try:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(os.path.realpath(sys.executable))
        else:
            application_path = os.path.dirname(os.path.realpath(__file__))
        if '~' in application_path and os_platform == 'windows':
            application_path = win32api.GetLongPathName(application_path)
        return application_path
    except Exception:
        print('Error while evaluation of application path')
        traceback.print_exc()
        if args.debug:
            sys.exit(1)

def is64bit():
    if False:
        while True:
            i = 10
    '\n    Checks if the system has a 64bit processor architecture\n    :return arch:\n    '
    return platform.machine().endswith('64')

def processExists(pid):
    if False:
        return 10
    '\n    Checks if a given process is running\n    :param pid:\n    :return:\n    '
    return psutil.pid_exists(pid)

def updateLoki(sigsOnly):
    if False:
        while True:
            i = 10
    logger.log('INFO', 'Update', 'Starting separate updater process ...')
    pArgs = []
    if os.path.exists(os.path.join(get_application_path(), 'loki-upgrader.exe')) and os_platform == 'windows':
        pArgs.append('loki-upgrader.exe')
    elif os.path.exists(os.path.join(get_application_path(), 'loki-upgrader.py')):
        pArgs.append(args.python)
        pArgs.append('loki-upgrader.py')
    else:
        logger.log('ERROR', 'Update', 'Cannot find neither thor-upgrader.exe nor thor-upgrader.py in the current working directory.')
    if sigsOnly:
        pArgs.append('--sigsonly')
        p = Popen(pArgs, shell=False)
        p.communicate()
    else:
        pArgs.append('--detached')
        Popen(pArgs, shell=False)

def walk_error(err):
    if False:
        print('Hello World!')
    if 'Error 3' in str(err):
        logging.error(str(err))
        print('Directory walk error')

def signal_handler(signal_name, frame):
    if False:
        print('Hello World!')
    try:
        print('------------------------------------------------------------------------------\n')
        logger.log('INFO', 'Init', "LOKI's work has been interrupted by a human. Returning to Asgard.")
    except Exception:
        print("LOKI's work has been interrupted by a human. Returning to Asgard.")
    sys.exit(0)

def main():
    if False:
        i = 10
        return i + 15
    '\n    Argument parsing function\n    :return:\n    '
    parser = argparse.ArgumentParser(description='Loki - Simple IOC Scanner')
    parser.add_argument('-p', help='Path to scan', metavar='path', default='C:\\')
    parser.add_argument('-s', help='Maximum file size to check in KB (default 5000 KB)', metavar='kilobyte', default=5000)
    parser.add_argument('-l', help='Log file', metavar='log-file', default='')
    parser.add_argument('-r', help='Remote syslog system', metavar='remote-loghost', default='')
    parser.add_argument('-t', help='Remote syslog port', metavar='remote-syslog-port', default=514)
    parser.add_argument('-a', help='Alert score', metavar='alert-level', default=100)
    parser.add_argument('-w', help='Warning score', metavar='warning-level', default=60)
    parser.add_argument('-n', help='Notice score', metavar='notice-level', default=40)
    parser.add_argument('--allhds', action='store_true', help='Scan all local hard drives (Windows only)', default=False)
    parser.add_argument('--alldrives', action='store_true', help='Scan all drives (including network drives and removable media)', default=False)
    parser.add_argument('--printall', action='store_true', help='Print all files that are scanned', default=False)
    parser.add_argument('--allreasons', action='store_true', help='Print all reasons that caused the score', default=False)
    parser.add_argument('--noprocscan', action='store_true', help='Skip the process scan', default=False)
    parser.add_argument('--nofilescan', action='store_true', help='Skip the file scan', default=False)
    parser.add_argument('--vulnchecks', action='store_true', help='Run the vulnerability checks', default=False)
    parser.add_argument('--nolevcheck', action='store_true', help='Skip the Levenshtein distance check', default=False)
    parser.add_argument('--scriptanalysis', action='store_true', help='Statistical analysis for scripts to detect obfuscated code (beta)', default=False)
    parser.add_argument('--rootkit', action='store_true', help='Skip the rootkit check', default=False)
    parser.add_argument('--noindicator', action='store_true', help='Do not show a progress indicator', default=False)
    parser.add_argument('--dontwait', action='store_true', help='Do not wait on exit', default=False)
    parser.add_argument('--intense', action='store_true', help='Intense scan mode (also scan unknown file types and all extensions)', default=False)
    parser.add_argument('--csv', action='store_true', help='Write CSV log format to STDOUT (machine processing)', default=False)
    parser.add_argument('--onlyrelevant', action='store_true', help='Only print warnings or alerts', default=False)
    parser.add_argument('--nolog', action='store_true', help="Don't write a local log file", default=False)
    parser.add_argument('--update', action='store_true', default=False, help='Update the signatures from the "signature-base" sub repository')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug output')
    parser.add_argument('--maxworkingset', type=int, default=200, help='Maximum working set size of processes to scan (in MB, default 100 MB)')
    parser.add_argument('--syslogtcp', action='store_true', default=False, help='Use TCP instead of UDP for syslog logging')
    parser.add_argument('--logfolder', help='Folder to use for logging when log file is not specified', metavar='log-folder', default='')
    parser.add_argument('--nopesieve', action='store_true', help='Do not perform pe-sieve scans', default=False)
    parser.add_argument('--pesieveshellc', action='store_true', help='Perform pe-sieve shellcode scan', default=False)
    parser.add_argument('--python', action='store', help='Override default python path', default='python')
    parser.add_argument('--nolisten', action='store_true', help='Dot not show listening connections', default=False)
    parser.add_argument('--excludeprocess', action='append', help='Specify an executable name to exclude from scans, can be used multiple times', default=[])
    parser.add_argument('--force', action='store_true', help="Force the scan on a certain folder (even if excluded with hard exclude in LOKI's code", default=False)
    parser.add_argument('--version', action='store_true', help='Shows welcome text and version of loki, then exit', default=False)
    args = parser.parse_args()
    if args.syslogtcp and (not args.r):
        print('Syslog logging set to TCP with --syslogtcp, but syslog logging not enabled with -r')
        sys.exit(1)
    if args.nolog and (args.l or args.logfolder):
        print('The --logfolder and -l directives are not compatible with --nolog')
        sys.exit(1)
    filename = 'loki_%s_%s.log' % (getHostname(os_platform), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.logfolder and args.l:
        print('Must specify either log folder with --logfolder, which uses the default filename, or log file with -l. Log file can be an absolute path')
        sys.exit(1)
    elif args.logfolder:
        args.logfolder = os.path.abspath(args.logfolder)
        args.l = os.path.join(args.logfolder, filename)
    elif not args.l:
        args.l = filename
    if args.nopesieve and args.pesieveshellc:
        print('The --pesieveshellc directive was specified, but pe-sieve scanning was disabled with --nopesieve')
        sys.exit(1)
    args.excludeprocess = [x.lower() for x in args.excludeprocess]
    return args
if __name__ == '__main__':
    signal_module.signal(signal_module.SIGINT, signal_handler)
    args = main()
    if os.path.exists(args.l):
        os.remove(args.l)
    LokiCustomFormatter = None
    logger = LokiLogger(args.nolog, args.l, getHostname(os_platform), args.r, int(args.t), args.syslogtcp, args.csv, args.onlyrelevant, args.debug, platform=os_platform, caller='main', customformatter=LokiCustomFormatter)
    if args.version:
        sys.exit(0)
    if args.update:
        updateLoki(sigsOnly=False)
        sys.exit(0)
    logger.log('NOTICE', 'Init', 'Starting Loki Scan VERSION: {3} SYSTEM: {0} TIME: {1} PLATFORM: {2}'.format(getHostname(os_platform), getSyslogTimestamp(), getPlatformFull(), logger.version))
    loki = Loki(args.intense)
    isAdmin = False
    if os_platform == 'windows':
        if shell.IsUserAnAdmin():
            isAdmin = True
            logger.log('INFO', 'Init', 'Current user has admin rights - very good')
        else:
            logger.log('NOTICE', 'Init', "Program should be run 'as Administrator' to ensure all access rights to process memory and file objects.")
    elif os.geteuid() == 0:
        isAdmin = True
        logger.log('INFO', 'Init', 'Current user is root - very good')
    else:
        logger.log('NOTICE', 'Init', "Program should be run as 'root' to ensure all access rights to process memory and file objects.")
    if os_platform == 'windows':
        setNice(logger)
    if args.rootkit and os_platform == 'windows':
        loki.check_rootkit()
    if args.vulnchecks and os_platform == 'windows':
        VChecker = VulnChecker(logger)
        VChecker.run()
    resultProc = False
    if not args.noprocscan and os_platform == 'windows':
        if isAdmin:
            loki.scan_processes(args.nopesieve, args.nolisten, args.excludeprocess, args.pesieveshellc)
        else:
            logger.log('NOTICE', 'Init', 'Skipping process memory check. User has no admin rights.')
    if not args.nofilescan:
        defaultPath = args.p
        if (os_platform == 'linux' or os_platform == 'macos') and args.p == 'C:\\':
            defaultPath = '/'
        if os_platform == 'windows':
            drives = win32api.GetLogicalDriveStrings().split('\x00')
            if args.allhds:
                for drive in drives:
                    if win32file.GetDriveType(drive) == win32file.DRIVE_FIXED:
                        loki.scan_path(drive)
            elif args.alldrives:
                for drive in drives:
                    loki.scan_path(drive)
            else:
                loki.scan_path(defaultPath)
        else:
            loki.scan_path(defaultPath)
    logger.log('NOTICE', 'Results', 'Results: {0} alerts, {1} warnings, {2} notices'.format(logger.alerts, logger.warnings, logger.notices))
    if logger.alerts:
        logger.log('RESULT', 'Results', 'Indicators detected!')
        logger.log('RESULT', 'Results', 'Loki recommends checking the elements on virustotal.com or Google and triage with a professional tool like THOR https://nextron-systems.com/thor in corporate networks.')
    elif logger.warnings:
        logger.log('RESULT', 'Results', 'Suspicious objects detected!')
        logger.log('RESULT', 'Results', 'Loki recommends a deeper analysis of the suspicious objects.')
    else:
        logger.log('RESULT', 'Results', 'SYSTEM SEEMS TO BE CLEAN.')
    logger.log('INFO', 'Results', 'Please report false positives via https://github.com/Neo23x0/signature-base')
    logger.log('NOTICE', 'Results', 'Finished LOKI Scan SYSTEM: %s TIME: %s' % (getHostname(os_platform), getSyslogTimestamp()))
sys.exit(0)