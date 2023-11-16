import sqlite3

def generate_it(filename):
    if False:
        return 10
    conn = sqlite3.connect(filename)
    conn.enable_load_extension(True)
    conn.load_extension('/usr/local/lib/mod_spatialite.dylib')
    conn.execute('select InitSpatialMetadata(1)')
    conn.executescript('create table museums (name text)')
    conn.execute("SELECT AddGeometryColumn('museums', 'point_geom', 4326, 'POINT', 2);")
    conn.execute('delete from spatial_ref_sys')
    conn.execute('delete from spatial_ref_sys_aux')
    conn.commit()
    conn.execute('vacuum')
    conn.close()
if __name__ == '__main__':
    generate_it('spatialite.db')