import time
import re
import gevent
from Config import config
from Db import Db
from util import helper

class BootstrapperDb(Db.Db):

    def __init__(self):
        if False:
            return 10
        self.version = 7
        self.hash_ids = {}
        super(BootstrapperDb, self).__init__({'db_name': 'Bootstrapper'}, '%s/bootstrapper.db' % config.data_dir)
        self.foreign_keys = True
        self.checkTables()
        self.updateHashCache()
        gevent.spawn(self.cleanup)

    def cleanup(self):
        if False:
            while True:
                i = 10
        while 1:
            time.sleep(4 * 60)
            timeout = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 60 * 40))
            self.execute('DELETE FROM peer WHERE date_announced < ?', [timeout])

    def updateHashCache(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.execute('SELECT * FROM hash')
        self.hash_ids = {row['hash']: row['hash_id'] for row in res}
        self.log.debug('Loaded %s hash_ids' % len(self.hash_ids))

    def checkTables(self):
        if False:
            print('Hello World!')
        version = int(self.execute('PRAGMA user_version').fetchone()[0])
        self.log.debug('Db version: %s, needed: %s' % (version, self.version))
        if version < self.version:
            self.createTables()
        else:
            self.execute('VACUUM')

    def createTables(self):
        if False:
            while True:
                i = 10
        self.execute('PRAGMA writable_schema = 1')
        self.execute("DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')")
        self.execute('PRAGMA writable_schema = 0')
        self.execute('VACUUM')
        self.execute('PRAGMA INTEGRITY_CHECK')
        self.execute('\n            CREATE TABLE peer (\n                peer_id        INTEGER PRIMARY KEY ASC AUTOINCREMENT NOT NULL UNIQUE,\n                type           TEXT,\n                address        TEXT,\n                port           INTEGER NOT NULL,\n                date_added     DATETIME DEFAULT (CURRENT_TIMESTAMP),\n                date_announced DATETIME DEFAULT (CURRENT_TIMESTAMP)\n            );\n        ')
        self.execute('CREATE UNIQUE INDEX peer_key ON peer (address, port);')
        self.execute('\n            CREATE TABLE peer_to_hash (\n                peer_to_hash_id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,\n                peer_id         INTEGER REFERENCES peer (peer_id) ON DELETE CASCADE,\n                hash_id         INTEGER REFERENCES hash (hash_id)\n            );\n        ')
        self.execute('CREATE INDEX peer_id ON peer_to_hash (peer_id);')
        self.execute('CREATE INDEX hash_id ON peer_to_hash (hash_id);')
        self.execute('\n            CREATE TABLE hash (\n                hash_id    INTEGER  PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,\n                hash       BLOB     UNIQUE NOT NULL,\n                date_added DATETIME DEFAULT (CURRENT_TIMESTAMP)\n            );\n        ')
        self.execute('PRAGMA user_version = %s' % self.version)

    def getHashId(self, hash):
        if False:
            while True:
                i = 10
        if hash not in self.hash_ids:
            self.log.debug('New hash: %s' % repr(hash))
            res = self.execute('INSERT OR IGNORE INTO hash ?', {'hash': hash})
            self.hash_ids[hash] = res.lastrowid
        return self.hash_ids[hash]

    def peerAnnounce(self, ip_type, address, port=None, hashes=[], onion_signed=False, delete_missing_hashes=False):
        if False:
            for i in range(10):
                print('nop')
        hashes_ids_announced = []
        for hash in hashes:
            hashes_ids_announced.append(self.getHashId(hash))
        res = self.execute('SELECT peer_id FROM peer WHERE ? LIMIT 1', {'address': address, 'port': port})
        user_row = res.fetchone()
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        if user_row:
            peer_id = user_row['peer_id']
            self.execute('UPDATE peer SET date_announced = ? WHERE peer_id = ?', (now, peer_id))
        else:
            self.log.debug('New peer: %s signed: %s' % (address, onion_signed))
            if ip_type == 'onion' and (not onion_signed):
                return len(hashes)
            res = self.execute('INSERT INTO peer ?', {'type': ip_type, 'address': address, 'port': port, 'date_announced': now})
            peer_id = res.lastrowid
        res = self.execute('SELECT * FROM peer_to_hash WHERE ?', {'peer_id': peer_id})
        hash_ids_db = [row['hash_id'] for row in res]
        if hash_ids_db != hashes_ids_announced:
            hash_ids_added = set(hashes_ids_announced) - set(hash_ids_db)
            hash_ids_removed = set(hash_ids_db) - set(hashes_ids_announced)
            if ip_type != 'onion' or onion_signed:
                for hash_id in hash_ids_added:
                    self.execute('INSERT INTO peer_to_hash ?', {'peer_id': peer_id, 'hash_id': hash_id})
                if hash_ids_removed and delete_missing_hashes:
                    self.execute('DELETE FROM peer_to_hash WHERE ?', {'peer_id': peer_id, 'hash_id': list(hash_ids_removed)})
            return len(hash_ids_added) + len(hash_ids_removed)
        else:
            return 0

    def peerList(self, hash, address=None, onions=[], port=None, limit=30, need_types=['ipv4', 'onion'], order=True):
        if False:
            print('Hello World!')
        back = {'ipv4': [], 'ipv6': [], 'onion': []}
        if limit == 0:
            return back
        hashid = self.getHashId(hash)
        if order:
            order_sql = 'ORDER BY date_announced DESC'
        else:
            order_sql = ''
        where_sql = 'hash_id = :hashid'
        if onions:
            onions_escaped = ["'%s'" % re.sub('[^a-z0-9,]', '', onion) for onion in onions if type(onion) is str]
            where_sql += ' AND address NOT IN (%s)' % ','.join(onions_escaped)
        elif address:
            where_sql += ' AND NOT (address = :address AND port = :port)'
        query = '\n            SELECT type, address, port\n            FROM peer_to_hash\n            LEFT JOIN peer USING (peer_id)\n            WHERE %s\n            %s\n            LIMIT :limit\n        ' % (where_sql, order_sql)
        res = self.execute(query, {'hashid': hashid, 'address': address, 'port': port, 'limit': limit})
        for row in res:
            if row['type'] in need_types:
                if row['type'] == 'onion':
                    packed = helper.packOnionAddress(row['address'], row['port'])
                else:
                    packed = helper.packAddress(str(row['address']), row['port'])
                back[row['type']].append(packed)
        return back