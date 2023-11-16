from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
workerTACTemplate = ["\nimport os\n\nfrom buildbot_worker.bot import Worker\nfrom twisted.application import service\n\nbasedir = %(basedir)r\nrotateLength = %(log-size)d\nmaxRotatedFiles = %(log-count)s\n\n# if this is a relocatable tac file, get the directory containing the TAC\nif basedir == '.':\n    import os.path\n    basedir = os.path.abspath(os.path.dirname(__file__))\n\n# note: this line is matched against to check that this is a worker\n# directory; do not edit it.\napplication = service.Application('buildbot-worker')\n", '\nfrom twisted.python.logfile import LogFile\nfrom twisted.python.log import ILogObserver, FileLogObserver\nlogfile = LogFile.fromFullPath(\n    os.path.join(basedir, "twistd.log"), rotateLength=rotateLength,\n    maxRotatedFiles=maxRotatedFiles)\napplication.setComponent(ILogObserver, FileLogObserver(logfile).emit)\n', '\nbuildmaster_host = %(host)r\nport = %(port)d\nconnection_string = None\n', '\nbuildmaster_host = None  # %(host)r\nport = None  # %(port)d\nconnection_string = %(connection-string)r\n', '\nworkername = %(name)r\npasswd = %(passwd)r\nkeepalive = %(keepalive)d\numask = %(umask)s\nmaxdelay = %(maxdelay)d\nnumcpus = %(numcpus)s\nallow_shutdown = %(allow-shutdown)r\nmaxretries = %(maxretries)s\nuse_tls = %(use-tls)s\ndelete_leftover_dirs = %(delete-leftover-dirs)s\nproxy_connection_string = %(proxy-connection-string)r\nprotocol = %(protocol)r\n\ns = Worker(buildmaster_host, port, workername, passwd, basedir,\n           keepalive, umask=umask, maxdelay=maxdelay,\n           numcpus=numcpus, allow_shutdown=allow_shutdown,\n           maxRetries=maxretries, protocol=protocol, useTls=use_tls,\n           delete_leftover_dirs=delete_leftover_dirs,\n           connection_string=connection_string,\n           proxy_connection_string=proxy_connection_string)\ns.setServiceParent(application)\n']

class CreateWorkerError(Exception):
    """
    Raised on errors while setting up worker directory.
    """

def _make_tac(config):
    if False:
        i = 10
        return i + 15
    if config['relocatable']:
        config['basedir'] = '.'
    workerTAC = [workerTACTemplate[0]]
    if not config['no-logrotate']:
        workerTAC.append(workerTACTemplate[1])
    if not config['connection-string']:
        workerTAC.append(workerTACTemplate[2])
    else:
        workerTAC.append(workerTACTemplate[3])
    workerTAC.extend(workerTACTemplate[4:])
    return ''.join(workerTAC) % config

def _makeBaseDir(basedir, quiet):
    if False:
        return 10
    "\n    Make worker base directory if needed.\n\n    @param basedir: worker base directory relative path\n    @param   quiet: if True, don't print info messages\n\n    @raise CreateWorkerError: on error making base directory\n    "
    if os.path.exists(basedir):
        if not quiet:
            print('updating existing installation')
        return
    if not quiet:
        print('mkdir', basedir)
    try:
        os.mkdir(basedir)
    except OSError as exception:
        raise CreateWorkerError('error creating directory {0}: {1}'.format(basedir, exception.strerror))

def _makeBuildbotTac(basedir, tac_file_contents, quiet):
    if False:
        print('Hello World!')
    "\n    Create buildbot.tac file. If buildbot.tac file already exists with\n    different contents, create buildbot.tac.new instead.\n\n    @param basedir: worker base directory relative path\n    @param tac_file_contents: contents of buildbot.tac file to write\n    @param quiet: if True, don't print info messages\n\n    @raise CreateWorkerError: on error reading or writing tac file\n    "
    tacfile = os.path.join(basedir, 'buildbot.tac')
    if os.path.exists(tacfile):
        try:
            with open(tacfile, 'rt') as f:
                oldcontents = f.read()
        except IOError as exception:
            raise CreateWorkerError('error reading {0}: {1}'.format(tacfile, exception.strerror))
        if oldcontents == tac_file_contents:
            if not quiet:
                print('buildbot.tac already exists and is correct')
            return
        if not quiet:
            print('not touching existing buildbot.tac')
            print('creating buildbot.tac.new instead')
        tacfile = os.path.join(basedir, 'buildbot.tac.new')
    try:
        with open(tacfile, 'wt') as f:
            f.write(tac_file_contents)
        os.chmod(tacfile, 384)
    except IOError as exception:
        raise CreateWorkerError('could not write {0}: {1}'.format(tacfile, exception.strerror))

def _makeInfoFiles(basedir, quiet):
    if False:
        i = 10
        return i + 15
    "\n    Create info/* files inside basedir.\n\n    @param basedir: worker base directory relative path\n    @param   quiet: if True, don't print info messages\n\n    @raise CreateWorkerError: on error making info directory or\n                             writing info files\n    "

    def createFile(path, file, contents):
        if False:
            i = 10
            return i + 15
        filepath = os.path.join(path, file)
        if os.path.exists(filepath):
            return False
        if not quiet:
            print('Creating {0}, you need to edit it appropriately.'.format(os.path.join('info', file)))
        try:
            open(filepath, 'wt').write(contents)
        except IOError as exception:
            raise CreateWorkerError('could not write {0}: {1}'.format(filepath, exception.strerror))
        return True
    path = os.path.join(basedir, 'info')
    if not os.path.exists(path):
        if not quiet:
            print('mkdir', path)
        try:
            os.mkdir(path)
        except OSError as exception:
            raise CreateWorkerError('error creating directory {0}: {1}'.format(path, exception.strerror))
    created = createFile(path, 'admin', 'Your Name Here <admin@youraddress.invalid>\n')
    created = createFile(path, 'host', 'Please put a description of this build host here\n')
    access_uri = os.path.join(path, 'access_uri')
    if not os.path.exists(access_uri):
        if not quiet:
            print('Not creating {0} - add it if you wish'.format(os.path.join('info', 'access_uri')))
    if created and (not quiet):
        print('Please edit the files in {0} appropriately.'.format(path))

def createWorker(config):
    if False:
        for i in range(10):
            print('nop')
    basedir = config['basedir']
    quiet = config['quiet']
    contents = _make_tac(config)
    try:
        _makeBaseDir(basedir, quiet)
        _makeBuildbotTac(basedir, contents, quiet)
        _makeInfoFiles(basedir, quiet)
    except CreateWorkerError as exception:
        print('{0}\nfailed to configure worker in {1}'.format(exception, config['basedir']))
        return 1
    if not quiet:
        print('worker configured in {0}'.format(basedir))
    return 0