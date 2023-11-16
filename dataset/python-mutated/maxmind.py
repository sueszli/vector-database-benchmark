"""This sub-module contains functions to interact with the MaxMind
files.

"""
import codecs
import os
import struct
import sys
from functools import partial, reduce
from multiprocessing import Pool
from ivre import config, utils
from ivre.db import DBData

class MaxMindFileIter:
    """Iterator for MaxMindFile"""

    def __init__(self, base):
        if False:
            return 10
        self.base = base
        self.current = []
        self.nextval = 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        if self.nextval is None:
            raise StopIteration()
        node_no = 0
        for i in range(96 if self.base.ip_version == 4 else 0, 128):
            try:
                flag = self.current[i]
            except IndexError:
                flag = self.nextval
                self.current.append(self.nextval)
                self.nextval = 0
            next_node_no = self.base.read_record(node_no, flag)
            if not next_node_no:
                raise Exception('Invalid file format')
            if next_node_no >= self.base.node_count:
                pos = next_node_no - self.base.node_count - self.base.DATA_SECTION_SEPARATOR_SIZE
                curvalinf = int(''.join((str(p) for p in self.current)) + '0' * (128 - len(self.current)), 2)
                curvalsup = int(''.join((str(p) for p in self.current)) + '1' * (128 - len(self.current)), 2)
                try:
                    while self.current.pop():
                        pass
                except IndexError:
                    self.nextval = None
                else:
                    self.current.append(1)
                return (curvalinf, curvalsup, self.base.decode(pos, self.base.data_section_start)[1])
            node_no = next_node_no
        raise StopIteration()

class EmptyMaxMindFile:
    """Stub to replace MaxMind databases parsers. Used when a file is
    missing to emit a warning message and return empty results.

    """

    def __init__(self, _):
        if False:
            while True:
                i = 10
        utils.LOGGER.warning('Cannot find Maxmind database files')

    @staticmethod
    def lookup(_):
        if False:
            return 10
        return {}

class MaxMindFile:
    """Parser for MaxMind databases.

    Code copied and adapted from
    https://github.com/yhirose/maxminddb/blob/master/lib/maxminddb.rb

    """
    METADATA_BEGIN_MARKER = b'\xab\xcd\xefMaxMind.com'
    DATA_SECTION_SEPARATOR_SIZE = 16
    SIZE_BASE_VALUES = [0, 29, 285, 65821]
    POINTER_BASE_VALUES = [0, 0, 2048, 526336]

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        self.path = path
        self._data = None
        pos = self.data.rindex(self.METADATA_BEGIN_MARKER) + len(self.METADATA_BEGIN_MARKER)
        metadata = self.metadata = self.decode(pos, 0)[1]
        self.ip_version = metadata['ip_version']
        self.node_count = metadata['node_count']
        self.node_byte_size = metadata['record_size'] * 2 // 8
        self.search_tree_size = self.node_count * self.node_byte_size
        self.data_section_start = self.search_tree_size + self.DATA_SECTION_SEPARATOR_SIZE

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        if self._data is None:
            with open(self.path, 'rb') as fdesc:
                self._data = fdesc.read()
        return self._data

    def read_byte(self, pos):
        if False:
            print('Hello World!')
        return self.data[pos]

    def read_value(self, pos, size):
        if False:
            return 10
        return reduce(lambda x, y: (x << 8) + y, struct.unpack('%dB' % size, self.data[pos:pos + size]), 0)

    def decode(self, pos, base_pos):
        if False:
            for i in range(10):
                print('nop')
        ctrl = self.data[pos + base_pos]
        pos += 1
        type_ = ctrl >> 5
        if type_ == 1:
            size = (ctrl >> 3 & 3) + 1
            val1 = ctrl & 7
            val2 = self.read_value(pos + base_pos, size)
            pointer = (val1 << 8 * size) + val2 + self.POINTER_BASE_VALUES[size]
            return (pos + size, self.decode(pointer, base_pos)[1])
        if not type_:
            type_ = 7 + self.read_byte(pos + base_pos)
            pos += 1
        size = ctrl & 31
        if size >= 29:
            byte_size = size - 29 + 1
            val = self.read_value(pos + base_pos, byte_size)
            pos += byte_size
            size = val + self.SIZE_BASE_VALUES[byte_size]
        if type_ == 2:
            val = self.data[pos + base_pos:pos + base_pos + size].decode('utf-8')
            pos += size
        elif type_ in [3, 15]:
            val = struct.unpack({3: '>d', 15: '>f'}[type_], self.data[pos + base_pos:pos + base_pos + size])[0]
            pos += size
        elif type_ == 4:
            val = self.data[pos + base_pos:pos + base_pos + size]
            pos += size
        elif type_ in [5, 6, 9, 10]:
            val = self.read_value(pos + base_pos, size)
            pos += size
        elif type_ == 7:
            val = {}
            for _ in range(size):
                (pos, k) = self.decode(pos, base_pos)
                (pos, v) = self.decode(pos, base_pos)
                val[k] = v
        elif type_ == 8:
            v1 = struct.unpack('>i', self.data[pos + base_pos:pos + base_pos + size])[0]
            bits = size * 8
            val = (v1 & ~(1 << bits)) - (v1 & 1 << bits)
            pos += size
        elif type_ == 11:
            val = []
            for _ in range(size):
                (pos, v) = self.decode(pos, base_pos)
                val.append(v)
        elif type_ == 12:
            raise Exception('TODO type == 12 (data cache container)')
        elif type_ == 13:
            val = None
        elif type_ == 14:
            val = bool(size)
        else:
            raise Exception('TODO type == %d (unknown)' % type_)
        return (pos, val)

    def read_record(self, node_no, flag):
        if False:
            while True:
                i = 10
        rec_byte_size = self.node_byte_size // 2
        pos = self.node_byte_size * node_no
        middle = self.read_byte(pos + rec_byte_size) if self.node_byte_size % 2 else 0
        if flag:
            val = self.read_value(pos + self.node_byte_size - rec_byte_size, rec_byte_size)
            val += (middle & 15) << 24 if middle else 0
        else:
            val = self.read_value(pos, rec_byte_size)
            val += (middle & 240) << 20 if middle else 0
        return val

    def __repr__(self):
        if False:
            return 10
        return '<%s from %s>' % (self.__class__.__name__, self.path)

    def lookup(self, ip):
        if False:
            for i in range(10):
                print('nop')
        node_no = 0
        addr = utils.force_ip2int(ip)
        for i in range(96 if self.ip_version == 4 else 0, 128):
            flag = addr >> 127 - i & 1
            next_node_no = self.read_record(node_no, flag)
            if not next_node_no:
                raise Exception('Invalid file format')
            if next_node_no >= self.node_count:
                pos = next_node_no - self.node_count - self.DATA_SECTION_SEPARATOR_SIZE
                return self.decode(pos, self.data_section_start)[1]
            node_no = next_node_no
        raise Exception('Invalid file format')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return MaxMindFileIter(self)

    @staticmethod
    def _get_fields(rec, fields):
        if False:
            print('Hello World!')
        for field in fields:
            val = rec
            for subfield in field.split('->'):
                try:
                    val = val[subfield]
                except TypeError:
                    try:
                        subfield = int(subfield)
                    except ValueError:
                        val = None
                        break
                    try:
                        val = val[subfield]
                    except IndexError:
                        val = None
                        break
                except KeyError:
                    val = None
                    break
            yield val

    def _get_ranges(self, fields):
        if False:
            print('Hello World!')
        gen = iter(self)
        try:
            (start, stop, rec) = next(gen)
        except StopIteration:
            return
        rec = tuple(self._get_fields(rec, fields))
        for (n_start, n_stop, n_rec) in gen:
            n_rec = tuple(self._get_fields(n_rec, fields))
            if n_start <= stop + 1 and n_rec == rec:
                stop = n_stop
            else:
                yield ((start, stop) + rec)
                (start, stop, rec) = (n_start, n_stop, n_rec)
        yield ((start, stop) + rec)

    def get_ranges(self, fields, cond=None):
        if False:
            while True:
                i = 10
        for rec in self._get_ranges(fields):
            if cond is None or cond(rec):
                yield rec

class MaxMindDBData(DBData):
    LANG = 'en'
    AS_KEYS = {'autonomous_system_number': 'as_num', 'autonomous_system_organization': 'as_name'}

    @property
    def db_asn(self):
        if False:
            return 10
        try:
            return self._db_asn
        except AttributeError:
            self._db_asn = EmptyMaxMindFile('ASN')
            return self._db_asn

    @property
    def db_city(self):
        if False:
            print('Hello World!')
        try:
            return self._db_city
        except AttributeError:
            self._db_city = EmptyMaxMindFile('City')
            return self._db_city

    @property
    def db_country(self):
        if False:
            while True:
                i = 10
        try:
            return self._db_country
        except AttributeError:
            self._db_country = EmptyMaxMindFile('Country')
            return self._db_country

    def __init__(self, url):
        if False:
            i = 10
            return i + 15
        self.basepath = url.path
        if sys.platform == 'win32' and self.basepath.startswith('/'):
            self.basepath = self.basepath[1:]
        self.reload_files()

    def reload_files(self):
        if False:
            while True:
                i = 10
        for fname in os.listdir(self.basepath):
            if fname.endswith('.mmdb'):
                subdb = MaxMindFile(os.path.join(self.basepath, fname))
                name = subdb.metadata['database_type'].lower()
                if name.startswith('geolite2-'):
                    name = name[9:]
                setattr(self, '_db_%s' % name, subdb)

    def as_byip(self, addr):
        if False:
            i = 10
            return i + 15
        return {self.AS_KEYS.get(key, key): value for (key, value) in self.db_asn.lookup(addr).items()}

    def location_byip(self, addr):
        if False:
            while True:
                i = 10
        raw = self.db_city.lookup(addr)
        result = {}
        sub = raw.get('subdivisions')
        if sub:
            result['region_code'] = tuple((v.get('iso_code') for v in sub))
            result['region_name'] = tuple((v.get('names', {}).get(self.LANG) for v in sub))
        sub = raw.get('continent')
        if sub:
            value = sub.get('code')
            if value:
                result['continent_code'] = value
            value = sub.get('names', {}).get(self.LANG)
            if value:
                result['continent_name'] = value
        sub = raw.get('country')
        if sub:
            value = sub.get('iso_code')
            if value:
                result['country_code'] = value
            value = sub.get('names', {}).get(self.LANG)
            if value:
                result['country_name'] = value
        sub = raw.get('registered_country')
        if sub:
            value = sub.get('iso_code')
            if value:
                result['registered_country_code'] = value
            value = sub.get('names', {}).get(self.LANG)
            if value:
                result['registered_country_name'] = value
        value = raw.get('city', {}).get('names', {}).get(self.LANG)
        if value:
            result['city'] = value
        value = raw.get('postal', {}).get('code')
        if value:
            result['postal_code'] = value
        sub = raw.get('location')
        if sub:
            try:
                result['coordinates'] = (sub['latitude'], sub['longitude'])
            except KeyError:
                pass
            value = sub.get('accuracy_radius')
            result['coordinates_accuracy_radius'] = value
        if result:
            return result
        return None

    def country_byip(self, addr):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        raw = self.db_country.lookup(addr)
        sub = raw.get('country')
        if sub:
            value = sub.get('iso_code')
            if value:
                result['country_code'] = value
            value = sub.get('names', {}).get(self.LANG)
            if value:
                result['country_name'] = value
        return result

    def dump_as_ranges(self, fdesc):
        if False:
            while True:
                i = 10
        for data in self.db_asn.get_ranges(['autonomous_system_number'], cond=lambda line: line[2] is not None):
            if data[0] > 4294967295:
                break
            fdesc.write('%d,%d,%d\n' % data)

    def dump_country_ranges(self, fdesc):
        if False:
            while True:
                i = 10
        for data in self.db_country.get_ranges(['country->iso_code'], cond=lambda line: line[2] is not None):
            if data[0] > 4294967295:
                break
            fdesc.write('%d,%d,%s\n' % data)

    def dump_registered_country_ranges(self, fdesc):
        if False:
            print('Hello World!')
        for data in self.db_country.get_ranges(['registered_country->iso_code'], cond=lambda line: line[2] is not None):
            if data[0] > 4294967295:
                break
            fdesc.write('%d,%d,%s\n' % data)

    def dump_city_ranges(self, fdesc):
        if False:
            while True:
                i = 10
        for data in self.db_city.get_ranges(['country->iso_code', 'subdivisions->0->iso_code', 'city->names->%s' % config.GEOIP_LANG, 'city->geoname_id'], cond=lambda line: line[2] is not None and (line[3] is not None or line[4] is not None)):
            if data[0] > 4294967295:
                break
            fdesc.write('%d,%d,%s,%s,%s,%s\n' % (data[:4] + (utils.encode_b64((data[4] or '').encode('utf-8')).decode('utf-8'),) + data[5:]))

    def build_dumps(self, force=False):
        if False:
            return 10
        'Produces CSV dump (.dump-IPv4.csv) files from Maxmind database\n        (.mmdb) files.\n\n        This function creates uses multiprocessing pool and makes several\n        calls to self._build_dump().\n\n        '
        with Pool() as pool:
            for _ in pool.imap(partial(self._build_dump, force), ['db_asn', 'db_country', 'db_registered_country', 'db_city'], chunksize=1):
                pass

    def _build_dump(self, force, attr):
        if False:
            while True:
                i = 10
        'Helper function used by MaxMindDBData.build_dumps() to create a\n        dump (.dump-IPv4.csv) file from a Maxmind database (.mmdb) file.\n\n        '
        dumper = {'db_asn': self.dump_as_ranges, 'db_country': self.dump_country_ranges, 'db_registered_country': self.dump_registered_country_ranges, 'db_city': self.dump_city_ranges}[attr]
        try:
            subdb = getattr(self, {'db_registered_country': 'db_country'}.get(attr, attr))
        except AttributeError:
            return
        if not subdb.path.endswith('.mmdb'):
            return
        csv_file = subdb.path[:-4] + 'dump-IPv4.csv'
        if attr == 'db_registered_country':
            if 'Country' not in csv_file:
                utils.LOGGER.error('Cannot build RegisteredCountry dump since filename %r does not contain "Country"', subdb.path)
            csv_file = csv_file.replace('Country', 'RegisteredCountry')
        if not force:
            mmdb_mtime = os.path.getmtime(subdb.path)
            try:
                csv_mtime = os.path.getmtime(csv_file)
            except OSError:
                pass
            else:
                if csv_mtime > mmdb_mtime:
                    utils.LOGGER.info('Skipping %r since %r is newer', os.path.basename(subdb.path), os.path.basename(csv_file))
                    return
        utils.LOGGER.info('Dumping %r to %r', os.path.basename(subdb.path), os.path.basename(csv_file))
        with codecs.open(csv_file, mode='w', encoding='utf-8') as fdesc:
            dumper(fdesc)