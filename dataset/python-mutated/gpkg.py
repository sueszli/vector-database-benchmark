import logging
log = logging.getLogger(__name__)
import os
import io
import math
import datetime
import sqlite3

class GeoPackage:
    MAX_DAYS = 90

    def __init__(self, path, tm):
        if False:
            return 10
        self.dbPath = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        (self.auth, self.code) = tm.CRS.split(':')
        self.code = int(self.code)
        self.tileSize = tm.tileSize
        (self.xmin, self.ymin, self.xmax, self.ymax) = tm.globalbbox
        self.resolutions = tm.getResList()
        if not self.isGPKG():
            self.create()
            self.insertMetadata()
            self.insertCRS(self.code, str(self.code), self.auth)
            self.insertTileMatrixSet()

    def isGPKG(self):
        if False:
            print('Hello World!')
        if not os.path.exists(self.dbPath):
            return False
        db = sqlite3.connect(self.dbPath)
        app_id = db.execute('PRAGMA application_id').fetchone()
        if not app_id[0] == 1196437808:
            db.close()
            return False
        try:
            db.execute('SELECT table_name FROM gpkg_contents LIMIT 1')
            db.execute('SELECT srs_name FROM gpkg_spatial_ref_sys LIMIT 1')
            db.execute('SELECT table_name FROM gpkg_tile_matrix_set LIMIT 1')
            db.execute('SELECT table_name FROM gpkg_tile_matrix LIMIT 1')
            db.execute('SELECT zoom_level, tile_column, tile_row, tile_data FROM gpkg_tiles LIMIT 1')
        except Exception as e:
            log.error('Incorrect GPKG schema', exc_info=True)
            db.close()
            return False
        else:
            db.close()
            return True

    def create(self):
        if False:
            print('Hello World!')
        'Create default geopackage schema on the database.'
        db = sqlite3.connect(self.dbPath)
        cursor = db.cursor()
        cursor.execute('PRAGMA application_id = 1196437808;')
        cursor.execute("\n\t\t\tCREATE TABLE gpkg_contents (\n\t\t\t\ttable_name TEXT NOT NULL PRIMARY KEY,\n\t\t\t\tdata_type TEXT NOT NULL,\n\t\t\t\tidentifier TEXT UNIQUE,\n\t\t\t\tdescription TEXT DEFAULT '',\n\t\t\t\tlast_change DATETIME NOT NULL DEFAULT\n\t\t\t\t(strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n\t\t\t\tmin_x DOUBLE,\n\t\t\t\tmin_y DOUBLE,\n\t\t\t\tmax_x DOUBLE,\n\t\t\t\tmax_y DOUBLE,\n\t\t\t\tsrs_id INTEGER,\n\t\t\t\tCONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)\n\t\t\t\t\tREFERENCES gpkg_spatial_ref_sys(srs_id));\n\t\t")
        cursor.execute('\n\t\t\tCREATE TABLE gpkg_spatial_ref_sys (\n\t\t\t\tsrs_name TEXT NOT NULL,\n\t\t\t\tsrs_id INTEGER NOT NULL PRIMARY KEY,\n\t\t\t\torganization TEXT NOT NULL,\n\t\t\t\torganization_coordsys_id INTEGER NOT NULL,\n\t\t\t\tdefinition TEXT NOT NULL,\n\t\t\t\tdescription TEXT);\n\t\t')
        cursor.execute('\n\t\t\tCREATE TABLE gpkg_tile_matrix_set (\n\t\t\t\ttable_name TEXT NOT NULL PRIMARY KEY,\n\t\t\t\tsrs_id INTEGER NOT NULL,\n\t\t\t\tmin_x DOUBLE NOT NULL,\n\t\t\t\tmin_y DOUBLE NOT NULL,\n\t\t\t\tmax_x DOUBLE NOT NULL,\n\t\t\t\tmax_y DOUBLE NOT NULL,\n\t\t\t\tCONSTRAINT fk_gtms_table_name FOREIGN KEY (table_name)\n\t\t\t\t\tREFERENCES gpkg_contents(table_name),\n\t\t\t\tCONSTRAINT fk_gtms_srs FOREIGN KEY (srs_id)\n\t\t\t\t\tREFERENCES gpkg_spatial_ref_sys(srs_id));\n\t\t')
        cursor.execute('\n\t\t\tCREATE TABLE gpkg_tile_matrix (\n\t\t\t\ttable_name TEXT NOT NULL,\n\t\t\t\tzoom_level INTEGER NOT NULL,\n\t\t\t\tmatrix_width INTEGER NOT NULL,\n\t\t\t\tmatrix_height INTEGER NOT NULL,\n\t\t\t\ttile_width INTEGER NOT NULL,\n\t\t\t\ttile_height INTEGER NOT NULL,\n\t\t\t\tpixel_x_size DOUBLE NOT NULL,\n\t\t\t\tpixel_y_size DOUBLE NOT NULL,\n\t\t\t\tCONSTRAINT pk_ttm PRIMARY KEY (table_name, zoom_level),\n\t\t\t\tCONSTRAINT fk_ttm_table_name FOREIGN KEY (table_name)\n\t\t\t\t\tREFERENCES gpkg_contents(table_name));\n\t\t')
        cursor.execute("\n\t\t\tCREATE TABLE gpkg_tiles (\n\t\t\t\tid INTEGER PRIMARY KEY AUTOINCREMENT,\n\t\t\t\tzoom_level INTEGER NOT NULL,\n\t\t\t\ttile_column INTEGER NOT NULL,\n\t\t\t\ttile_row INTEGER NOT NULL,\n\t\t\t\ttile_data BLOB NOT NULL,\n\t\t\t\tlast_modified TIMESTAMP DEFAULT (datetime('now','localtime')),\n\t\t\t\tUNIQUE (zoom_level, tile_column, tile_row));\n\t\t")
        db.close()

    def insertMetadata(self):
        if False:
            print('Hello World!')
        db = sqlite3.connect(self.dbPath)
        query = 'INSERT INTO gpkg_contents (\n\t\t\t\t\ttable_name, data_type,\n\t\t\t\t\tidentifier, description,\n\t\t\t\t\tmin_x, min_y, max_x, max_y,\n\t\t\t\t\tsrs_id)\n\t\t\t\tVALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);'
        db.execute(query, ('gpkg_tiles', 'tiles', self.name, 'Created with BlenderGIS', self.xmin, self.ymin, self.xmax, self.ymax, self.code))
        db.commit()
        db.close()

    def insertCRS(self, code, name, auth='EPSG', wkt=''):
        if False:
            print('Hello World!')
        db = sqlite3.connect(self.dbPath)
        db.execute(' INSERT INTO gpkg_spatial_ref_sys (\n\t\t\t\t\tsrs_id,\n\t\t\t\t\torganization,\n\t\t\t\t\torganization_coordsys_id,\n\t\t\t\t\tsrs_name,\n\t\t\t\t\tdefinition)\n\t\t\t\tVALUES (?, ?, ?, ?, ?)\n\t\t\t', (code, auth, code, name, wkt))
        db.commit()
        db.close()

    def insertTileMatrixSet(self):
        if False:
            for i in range(10):
                print('nop')
        db = sqlite3.connect(self.dbPath)
        query = 'INSERT OR REPLACE INTO gpkg_tile_matrix_set (\n\t\t\t\t\ttable_name, srs_id,\n\t\t\t\t\tmin_x, min_y, max_x, max_y)\n\t\t\t\tVALUES (?, ?, ?, ?, ?, ?);'
        db.execute(query, ('gpkg_tiles', self.code, self.xmin, self.ymin, self.xmax, self.ymax))
        for (level, res) in enumerate(self.resolutions):
            w = math.ceil((self.xmax - self.xmin) / (self.tileSize * res))
            h = math.ceil((self.ymax - self.ymin) / (self.tileSize * res))
            query = 'INSERT OR REPLACE INTO gpkg_tile_matrix (\n\t\t\t\t\t\ttable_name, zoom_level,\n\t\t\t\t\t\tmatrix_width, matrix_height,\n\t\t\t\t\t\ttile_width, tile_height,\n\t\t\t\t\t\tpixel_x_size, pixel_y_size)\n\t\t\t\t\tVALUES (?, ?, ?, ?, ?, ?, ?, ?);'
            db.execute(query, ('gpkg_tiles', level, w, h, self.tileSize, self.tileSize, res, res))
        db.commit()
        db.close()

    def hasTile(self, x, y, z):
        if False:
            return 10
        if self.getTile(x, y, z) is not None:
            return True
        else:
            return False

    def getTile(self, x, y, z):
        if False:
            i = 10
            return i + 15
        'return tilde_data if tile exists otherwie return None'
        db = sqlite3.connect(self.dbPath, detect_types=sqlite3.PARSE_DECLTYPES)
        query = 'SELECT tile_data, last_modified FROM gpkg_tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?'
        result = db.execute(query, (z, x, y)).fetchone()
        db.close()
        if result is None:
            return None
        timeDelta = datetime.datetime.now() - result[1]
        if timeDelta.days > self.MAX_DAYS:
            return None
        return result[0]

    def putTile(self, x, y, z, data):
        if False:
            while True:
                i = 10
        db = sqlite3.connect(self.dbPath)
        query = 'INSERT OR REPLACE INTO gpkg_tiles\n\t\t(tile_column, tile_row, zoom_level, tile_data) VALUES (?,?,?,?)'
        db.execute(query, (x, y, z, data))
        db.commit()
        db.close()

    def listExistingTiles(self, tiles):
        if False:
            print('Hello World!')
        '\n\t\tinput : tiles list [(x,y,z)]\n\t\toutput : tiles list set [(x,y,z)] of existing records in cache db'
        db = sqlite3.connect(self.dbPath, detect_types=sqlite3.PARSE_DECLTYPES)
        (x, y, z) = zip(*tiles)
        query = 'SELECT tile_column, tile_row, zoom_level FROM gpkg_tiles WHERE julianday() - julianday(last_modified) < ?AND zoom_level BETWEEN ? AND ? AND tile_column BETWEEN ? AND ? AND tile_row BETWEEN ? AND ?'
        result = db.execute(query, (GeoPackage.MAX_DAYS, min(z), max(z), min(x), max(x), min(y), max(y))).fetchall()
        db.close()
        return set(result)

    def listMissingTiles(self, tiles):
        if False:
            print('Hello World!')
        existing = self.listExistingTiles(tiles)
        return set(tiles) - existing

    def getTiles(self, tiles):
        if False:
            while True:
                i = 10
        'tiles = list of (x,y,z) tuple\n\t\treturn list of (x,y,z,data) tuple'
        db = sqlite3.connect(self.dbPath, detect_types=sqlite3.PARSE_DECLTYPES)
        (x, y, z) = zip(*tiles)
        query = 'SELECT tile_column, tile_row, zoom_level, tile_data FROM gpkg_tiles WHERE julianday() - julianday(last_modified) < ?AND zoom_level BETWEEN ? AND ? AND tile_column BETWEEN ? AND ? AND tile_row BETWEEN ? AND ?'
        result = db.execute(query, (GeoPackage.MAX_DAYS, min(z), max(z), min(x), max(x), min(y), max(y))).fetchall()
        db.close()
        return result

    def putTiles(self, tiles):
        if False:
            return 10
        'tiles = list of (x,y,z,data) tuple'
        db = sqlite3.connect(self.dbPath)
        query = 'INSERT OR REPLACE INTO gpkg_tiles\n\t\t(tile_column, tile_row, zoom_level, tile_data) VALUES (?,?,?,?)'
        db.executemany(query, tiles)
        db.commit()
        db.close()