"""
stream changes in mysql (on the torrents and statistics table) into
elasticsearch as they happen on the binlog. This keeps elasticsearch in sync
with whatever you do to the database, including stuff like admin queries. Also,
because mysql keeps the binlog around for N days before deleting old stuff, you
can survive a hiccup of elasticsearch or this script dying and pick up where
you left off.

For that "picking up" part, this script depends on one piece of external state:
its last known binlog filename and position. This is saved off as a JSON file
to a configurable location on the filesystem periodically. If the file is not
present then you can initialize it with the values from `SHOW MASTER STATUS`
from the mysql repl, which will start the sync from current state.

In the case of catastrophic elasticsearch meltdown where you need to
reconstruct the index, you'll want to be a bit careful with coordinating
sync_es and import_to_es scripts. If you run import_to_es first than run
sync_es against SHOW MASTER STATUS, anything that changed the database between
when import_to_es and sync_es will be lost. Instead, you can run SHOW MASTER
STATUS _before_ you run import_to_es. That way you'll definitely pick up any
changes that happen while the import_to_es script is dumping stuff from the
database into es, at the expense of redoing a (small) amount of indexing.

This uses multithreading so we don't have to block on socket io (both binlog
reading and es POSTing). asyncio soon™

This script will exit on any sort of exception, so you'll want to use your
supervisor's restart functionality, e.g. Restart=failure in systemd, or
the poor man's `while true; do sync_es.py; sleep 1; done` in tmux.
"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import UpdateRowsEvent, DeleteRowsEvent, WriteRowsEvent
from datetime import datetime
from nyaa import create_app, db, models
from nyaa.models import TorrentFlags
app = create_app('config')
import sys
import json
import time
import logging
from statsd import StatsClient
from threading import Thread
from queue import Queue, Empty
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s - %(message)s')
log = logging.getLogger('sync_es')
log.setLevel(logging.INFO)
if len(sys.argv) != 2:
    print('need config.json location', file=sys.stderr)
    sys.exit(-1)
with open(sys.argv[1]) as f:
    config = json.load(f)
stats = StatsClient('localhost', 8125, prefix='sync_es')
SAVE_LOC = config.get('save_loc', '/tmp/pos.json')
MYSQL_HOST = config.get('mysql_host', '127.0.0.1')
MYSQL_PORT = config.get('mysql_port', 3306)
MYSQL_USER = config.get('mysql_user', 'root')
MYSQL_PW = config.get('mysql_password', 'dunnolol')
NT_DB = config.get('database', 'nyaav2')
INTERNAL_QUEUE_DEPTH = config.get('internal_queue_depth', 10000)
ES_CHUNK_SIZE = config.get('es_chunk_size', 10000)
FLUSH_INTERVAL = config.get('flush_interval', 5)

def pad_bytes(in_bytes, size):
    if False:
        return 10
    return in_bytes + b'\x00' * max(0, size - len(in_bytes))

def reindex_torrent(t, index_name):
    if False:
        while True:
            i = 10
    f = t['flags']
    doc = {'id': t['id'], 'display_name': t['display_name'], 'created_time': t['created_time'], 'updated_time': t['updated_time'], 'description': t['description'], 'info_hash': pad_bytes(t['info_hash'], 20).hex(), 'filesize': t['filesize'], 'uploader_id': t['uploader_id'], 'main_category_id': t['main_category_id'], 'sub_category_id': t['sub_category_id'], 'comment_count': t['comment_count'], 'anonymous': bool(f & TorrentFlags.ANONYMOUS), 'trusted': bool(f & TorrentFlags.TRUSTED), 'remake': bool(f & TorrentFlags.REMAKE), 'complete': bool(f & TorrentFlags.COMPLETE), 'hidden': bool(f & TorrentFlags.HIDDEN), 'deleted': bool(f & TorrentFlags.DELETED), 'has_torrent': bool(t['has_torrent'])}
    return {'_op_type': 'update', '_index': index_name, '_id': str(t['id']), 'doc': doc, 'doc_as_upsert': True}

def reindex_stats(s, index_name):
    if False:
        i = 10
        return i + 15
    return {'_op_type': 'update', '_index': index_name, '_id': str(s['torrent_id']), 'doc': {'stats_last_updated': s['last_updated'], 'download_count': s['download_count'], 'leech_count': s['leech_count'], 'seed_count': s['seed_count']}}

def delet_this(row, index_name):
    if False:
        print('Hello World!')
    return {'_op_type': 'delete', '_index': index_name, '_id': str(row['values']['id'])}

class ExitingThread(Thread):

    def run(self):
        if False:
            return 10
        try:
            self.run_happy()
        except:
            log.exception('something happened')
            import os
            os._exit(1)

class BinlogReader(ExitingThread):

    def __init__(self, write_buf):
        if False:
            for i in range(10):
                print('nop')
        Thread.__init__(self)
        self.write_buf = write_buf

    def run_happy(self):
        if False:
            print('Hello World!')
        with open(SAVE_LOC) as f:
            pos = json.load(f)
        stream = BinLogStreamReader(connection_settings={'host': MYSQL_HOST, 'port': MYSQL_PORT, 'user': MYSQL_USER, 'passwd': MYSQL_PW}, server_id=10, only_schemas=[NT_DB], only_tables=['nyaa_torrents', 'nyaa_statistics', 'sukebei_torrents', 'sukebei_statistics'], resume_stream=True, log_file=pos['log_file'], log_pos=pos['log_pos'], only_events=[UpdateRowsEvent, DeleteRowsEvent, WriteRowsEvent], blocking=True)
        log.info(f'reading binlog from {stream.log_file}/{stream.log_pos}')
        for event in stream:
            pos = (stream.log_file, stream.log_pos, event.timestamp)
            with stats.pipeline() as s:
                s.incr('total_events')
                s.incr(f'event.{event.table}.{type(event).__name__}')
                s.incr('total_rows', len(event.rows))
                s.incr(f'rows.{event.table}.{type(event).__name__}', len(event.rows))
                s.timing(f'rows_per_event.{event.table}.{type(event).__name__}', len(event.rows))
            if event.table == 'nyaa_torrents' or event.table == 'sukebei_torrents':
                if event.table == 'nyaa_torrents':
                    index_name = 'nyaa'
                else:
                    index_name = 'sukebei'
                if type(event) is WriteRowsEvent:
                    for row in event.rows:
                        self.write_buf.put((pos, reindex_torrent(row['values'], index_name)), block=True)
                elif type(event) is UpdateRowsEvent:
                    for row in event.rows:
                        self.write_buf.put((pos, reindex_torrent(row['after_values'], index_name)), block=True)
                elif type(event) is DeleteRowsEvent:
                    for row in event.rows:
                        self.write_buf.put((pos, delet_this(row, index_name)), block=True)
                else:
                    raise Exception(f'unknown event {type(event)}')
            elif event.table == 'nyaa_statistics' or event.table == 'sukebei_statistics':
                if event.table == 'nyaa_statistics':
                    index_name = 'nyaa'
                else:
                    index_name = 'sukebei'
                if type(event) is WriteRowsEvent:
                    for row in event.rows:
                        self.write_buf.put((pos, reindex_stats(row['values'], index_name)), block=True)
                elif type(event) is UpdateRowsEvent:
                    for row in event.rows:
                        self.write_buf.put((pos, reindex_stats(row['after_values'], index_name)), block=True)
                elif type(event) is DeleteRowsEvent:
                    pass
                else:
                    raise Exception(f'unknown event {type(event)}')
            else:
                raise Exception(f'unknown table {s.table}')

class EsPoster(ExitingThread):

    def __init__(self, read_buf, chunk_size=1000, flush_interval=5):
        if False:
            return 10
        Thread.__init__(self)
        self.read_buf = read_buf
        self.chunk_size = chunk_size
        self.flush_interval = flush_interval

    def run_happy(self):
        if False:
            return 10
        es = Elasticsearch(hosts=app.config['ES_HOSTS'], timeout=30)
        last_save = time.time()
        since_last = 0
        posted_log_file = None
        posted_log_pos = None
        while True:
            actions = []
            now = time.time()
            deadline = now + self.flush_interval
            while len(actions) < self.chunk_size and now < deadline:
                timeout = deadline - now
                try:
                    ((log_file, log_pos, timestamp), action) = self.read_buf.get(block=True, timeout=timeout)
                    actions.append(action)
                    now = time.time()
                except Empty:
                    break
            if actions:
                stats.timing('actions_per_bulk', len(actions))
                try:
                    with stats.timer('post_bulk'):
                        bulk(es, actions, chunk_size=self.chunk_size)
                except BulkIndexError as bie:
                    for e in bie.errors:
                        try:
                            if e['update']['error']['type'] != 'document_missing_exception':
                                raise bie
                        except KeyError:
                            raise bie
                posted_log_file = log_file
                posted_log_pos = log_pos
                stats.gauge('process_latency', int((time.time() - timestamp) * 1000))
            else:
                log.debug('no changes...')
            since_last += len(actions)
            if posted_log_file is not None and (since_last >= 10000 or time.time() - last_save > 10):
                log.info(f'saving position {log_file}/{log_pos}, {time.time() - timestamp:,.3f} seconds behind')
                with stats.timer('save_pos'):
                    with open(SAVE_LOC, 'w') as f:
                        json.dump({'log_file': posted_log_file, 'log_pos': posted_log_pos}, f)
                last_save = time.time()
                since_last = 0
                posted_log_file = None
                posted_log_pos = None
buf = Queue(maxsize=INTERNAL_QUEUE_DEPTH)
reader = BinlogReader(buf)
reader.daemon = True
writer = EsPoster(buf, chunk_size=ES_CHUNK_SIZE, flush_interval=FLUSH_INTERVAL)
writer.daemon = True
reader.start()
writer.start()
while True:
    stats.gauge('queue_depth', buf.qsize())
    time.sleep(1)