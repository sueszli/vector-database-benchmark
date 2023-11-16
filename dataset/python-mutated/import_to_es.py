"""
Bulk load torents from mysql into elasticsearch `nyaav2` index,
which is assumed to already exist.
This is a one-shot deal, so you'd either need to complement it
with a cron job or some binlog-reading thing (TODO)
"""
import sys
import json
import progressbar
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch import helpers
from nyaa import create_app, models
from nyaa.extensions import db
app = create_app('config')
es = Elasticsearch(hosts=app.config['ES_HOSTS'], timeout=30)
ic = IndicesClient(es)

def pad_bytes(in_bytes, size):
    if False:
        print('Hello World!')
    return in_bytes + b'\x00' * max(0, size - len(in_bytes))

def mk_es(t, index_name):
    if False:
        i = 10
        return i + 15
    return {'_id': t.id, '_index': index_name, '_source': {'id': t.id, 'display_name': t.display_name, 'created_time': t.created_time, 'info_hash': pad_bytes(t.info_hash, 20).hex(), 'filesize': t.filesize, 'uploader_id': t.uploader_id, 'main_category_id': t.main_category_id, 'sub_category_id': t.sub_category_id, 'comment_count': t.comment_count, 'anonymous': bool(t.anonymous), 'trusted': bool(t.trusted), 'remake': bool(t.remake), 'complete': bool(t.complete), 'hidden': bool(t.hidden), 'deleted': bool(t.deleted), 'has_torrent': t.has_torrent, 'download_count': t.stats.download_count, 'leech_count': t.stats.leech_count, 'seed_count': t.stats.seed_count}}

def page_query(query, limit=sys.maxsize, batch_size=10000, progress_bar=None):
    if False:
        i = 10
        return i + 15
    start = 0
    while True:
        stop = min(limit, start + batch_size)
        if stop == start:
            break
        things = query.slice(start, stop)
        if not things:
            break
        had_things = False
        for thing in things:
            had_things = True
            yield thing
        if not had_things or stop == limit:
            break
        if progress_bar:
            progress_bar.update(start)
        start = min(limit, start + batch_size)
FLAVORS = [('nyaa', models.NyaaTorrent), ('sukebei', models.SukebeiTorrent)]
with app.app_context():
    master_status = db.engine.execute('SHOW MASTER STATUS;').fetchone()
    position_json = {'log_file': master_status[0], 'log_pos': master_status[1]}
    print('Save the following in the file configured in your ES sync config JSON:')
    print(json.dumps(position_json))
    for (flavor, torrent_class) in FLAVORS:
        print('Importing torrents for index', flavor, 'from', torrent_class)
        bar = progressbar.ProgressBar(maxval=torrent_class.query.count(), widgets=[progressbar.SimpleProgress(), ' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        ic.put_settings(body={'index': {'refresh_interval': '-1'}}, index=flavor)
        bar.start()
        helpers.bulk(es, (mk_es(t, flavor) for t in page_query(torrent_class.query, progress_bar=bar)), chunk_size=10000)
        bar.finish()
        ic.refresh(index=flavor)
        print('Index refresh done.')
        ic.put_settings(body={'index': {'refresh_interval': '30s'}}, index=flavor)