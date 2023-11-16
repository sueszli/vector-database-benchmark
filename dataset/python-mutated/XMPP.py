import asyncio
import os
import ssl
import slixmpp
from slixmpp.xmlstream.handler import Callback
from slixmpp.xmlstream.matcher import MatchXPath
from ..base.chat_bot import ChatBot

class XMPP(ChatBot):
    __name__ = 'XMPP'
    __type__ = 'addon'
    __version__ = '0.23'
    __status__ = 'testing'
    __config__ = [('enabled', 'bool', 'Activated', False), ('jid', 'str', 'Jabber ID', 'user@exmaple-jabber-server.org'), ('pw', 'str', 'Password', ''), ('use_ipv6', 'bool', 'Use ipv6', False), ('tls', 'bool', 'Use TLS', True), ('use_ssl', 'bool', 'Use old SSL', False), ('owners', 'str', 'List of JIDs accepting commands from', 'me@icq-gateway.org;some@msn-gateway.org'), ('captcha', 'bool', 'Send captcha requests', True), ('info_file', 'bool', 'Inform about every file finished', False), ('info_pack', 'bool', 'Inform about every package finished', True), ('all_download', 'bool', 'Inform about all download finished', False), ('package_failed', 'bool', 'Notify package failed', False), ('download_failed', 'bool', 'Notify download failed', True), ('download_start', 'bool', 'Notify download start', True), ('maxline', 'int', 'Maximum line per message', 6)]
    __description__ = 'Connect to jabber and let owner perform different tasks'
    __license__ = 'GPLv3'
    __authors__ = [('RaNaN', 'RaNaN@pyload.net'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]

    def activate(self):
        if False:
            return 10
        self.log_debug('activate')
        self.jid = slixmpp.jid.JID(self.config.get('jid'))
        self.jid.resource = 'PyLoadNotifyBot'
        self.log_debug(self.jid)
        super().activate()

    def run(self):
        if False:
            return 10
        self.log_debug('def run')
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        xmpp = XMPPClient(self.jid, self.config.get('pw'), self.log_info, self.log_debug)
        self.log_debug('activate xmpp')
        xmpp.use_ipv6 = self.config.get('use_ipv6')
        xmpp.register_plugin('xep_0030')
        xmpp.register_plugin('xep_0004')
        xmpp.register_plugin('xep_0060')
        xmpp.register_plugin('xep_0199')
        xmpp.ssl_version = ssl.PROTOCOL_TLSv1_2
        xmpp.add_event_handler('message', self.message)
        xmpp.add_event_handler('connected', self.connected)
        xmpp.add_event_handler('connection_failed', self.connection_failed)
        xmpp.add_event_handler('disconnected', self.disconnected)
        xmpp.add_event_handler('failed_auth', self.failed_auth)
        xmpp.add_event_handler('changed_status', self.changed_status)
        xmpp.add_event_handler('presence_error', self.presence_error)
        xmpp.add_event_handler('presence_unavailable', self.presence_unavailable)
        xmpp.register_handler(Callback('Stream Error', MatchXPath(f'{{{xmpp.stream_ns}}}error'), self.stream_error))
        self.xmpp = xmpp
        self.xmpp.connect(use_ssl=self.config.get('use_ssl'), force_starttls=self.config.get('tls'))
        self.xmpp.process(forever=True)

    def changed_status(self, stanza=None):
        if False:
            for i in range(10):
                print('nop')
        self.log_debug('changed_status', stanza, stanza.get_type())

    def connection_failed(self, stanza=None):
        if False:
            i = 10
            return i + 15
        self.log_error('Unable to connect', stanza)

    def connected(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.log_info('Client was connected', event)

    def disconnected(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.log_info('Client was disconnected', event)

    def presence_error(self, stanza=None):
        if False:
            print('Hello World!')
        self.log_debug('presence_error', stanza)

    def presence_unavailable(self, stanza=None):
        if False:
            return 10
        self.log_debug('presence_unavailable', stanza)

    def failed_auth(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.log_info('Failed to authenticate')

    def stream_error(self, err=None):
        if False:
            print('Hello World!')
        self.log_debug('Stream Error', err)

    def message(self, stanza):
        if False:
            return 10
        '\n        Message handler for the component.\n        '
        self.log_debug('message', stanza)
        subject = stanza['subject']
        body = stanza['body']
        msg_type = stanza['type']
        sender_jid = stanza['from']
        names = self.config.get('owners').split(';')
        self.log_debug(f'Message from {sender_jid} received.')
        self.log_debug(f'Body: {body} Subject: {subject} Type: {msg_type}')
        if msg_type == 'headline':
            return True
        if subject:
            subject = 'Re: ' + subject
        if not (sender_jid.username in names or sender_jid.bare in names):
            return True
        temp = body.split()
        try:
            command = temp[0]
            args = temp[1:]
        except IndexError:
            command = 'error'
            args = []
        ret = False
        try:
            res = self.do_bot_command(command, args)
            if res:
                msg_reply = '\n'.join(res)
            else:
                msg_reply = 'ERROR: invalid command, enter: help'
            self.log_debug('Send response')
            ret = stanza.reply(msg_reply).send()
        except Exception as exc:
            self.log_error(exc)
            stanza.reply('ERROR: ' + str(exc)).send()
        return ret

    def announce(self, message):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send message to all owners\n        '
        self.log_debug('Announce, message:', message)
        for user in self.config.get('owners').split(';'):
            self.log_debug('Send message to', user)
            to_jid = slixmpp.jid.JID(user)
            self.xmpp.sendMessage(mfrom=self.jid, mto=to_jid, mtype='chat', mbody=str(message))

    def exit(self):
        if False:
            return 10
        self.xmpp.disconnect()

    def before_reconnect(self, ip):
        if False:
            while True:
                i = 10
        self.log_debug('before_reconnect')
        self.xmpp.disconnect()

    def after_reconnect(self, ip, oldip):
        if False:
            print('Hello World!')
        self.log_debug('after_reconnect')
        self.xmpp.connect()

    def download_failed(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        self.log_debug('download_failed', pyfile, pyfile.error)
        try:
            if self.config.get('download_failed'):
                self.announce(self._('Download failed: {} (#{}) in #{} @ {}: {}').format(pyfile.name, pyfile.id, pyfile.packageid, pyfile.pluginname, pyfile.error))
        except Exception as exc:
            self.log_error(exc)

    def package_failed(self, pypack):
        if False:
            while True:
                i = 10
        self.log_debug('package_failed', pypack)
        try:
            if self.config.get('package_failed'):
                self.announce(self._('Package failed: {} ({}).').format(pypack.name, pypack.id))
        except Exception as exc:
            self.log_error(exc)

    def package_finished(self, pypack):
        if False:
            print('Hello World!')
        self.log_debug('package_finished')
        try:
            if self.config.get('info_pack'):
                self.announce(self._('Package finished: {} ({}).').format(pypack.name, pypack.id))
        except Exception as exc:
            self.log_error(exc)

    def download_finished(self, pyfile):
        if False:
            print('Hello World!')
        self.log_debug('download_finished')
        try:
            if self.config.get('info_file'):
                self.announce(self._('Download finished: {} (#{}) in #{} @ {}').format(pyfile.name, pyfile.id, pyfile.packageid, pyfile.pluginname))
        except Exception as exc:
            self.log_error(exc)

    def all_downloads_processed(self):
        if False:
            while True:
                i = 10
        self.log_debug('all_downloads_processed')
        try:
            if self.config.get('all_download'):
                self.announce(self._('All download finished.'))
        except Exception:
            pass

    def download_start(self, pyfile, url, filename):
        if False:
            while True:
                i = 10
        self.log_debug('download_start', pyfile, url, filename)
        try:
            if self.config.get('download_start'):
                self.announce(self._('Download start: {} (#{}) in (#{}) @ {}.').format(pyfile.name, pyfile.id, pyfile.packageid, pyfile.pluginname))
        except Exception:
            pass

class XMPPClient(slixmpp.ClientXMPP):

    def __init__(self, jid, password, log_info, log_debug):
        if False:
            i = 10
            return i + 15
        self.log_debug = log_debug
        self.log_info = log_info
        slixmpp.ClientXMPP.__init__(self, jid, password)
        self.add_event_handler('session_start', self.start)

    def start(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.log_debug('Session started')
        self.send_presence()
        self.get_roster(timeout=60)