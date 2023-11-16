from app.helpers import asset_site, asset_wih
from app.helpers.scope import get_scope_by_scope_id
from app.services import run_wih
from app.utils import get_logger, check_domain_black
from app.modules import WihRecord
from app import utils
logger = get_logger()

class AssetWihMonitor(object):

    def __init__(self, scope_id: str):
        if False:
            print('Hello World!')
        self.scope_id = scope_id
        self.scope_domains = []
        self.scope_name = None
        self.sites = []
        self._wih_record_fnv_hash = None

    def init_scope_data(self):
        if False:
            print('Hello World!')
        scope_data = get_scope_by_scope_id(self.scope_id)
        if not scope_data:
            raise Exception('没有找到资产组 {}'.format(self.scope_id))
        self.scope_name = scope_data.get('name', '')
        scope_type = scope_data.get('scope_type', '')
        if scope_type == 'domain':
            self.scope_domains = scope_data.get('scope_array', [])
        self.sites = asset_site.find_site_by_scope_id(self.scope_id)

    def have_asset_wih_record(self, record: WihRecord) -> bool:
        if False:
            while True:
                i = 10
        '\n        检查数据库中是否已经存在记录\n        :param record:\n        :return:\n        '
        query = {'scope_id': self.scope_id, 'fnv_hash': str(record.fnv_hash)}
        item = utils.conn_db('asset_wih').find_one(query)
        if item:
            return True
        return False

    def save_asset_wih_record(self, record: WihRecord):
        if False:
            return 10
        '\n        保存到数据库\n        :param record: \n        :return: \n        '
        if self.have_asset_wih_record(record):
            return
        item = record.dump_json()
        item['scope_id'] = self.scope_id
        curr_date = utils.curr_date_obj()
        item['save_date'] = curr_date
        item['update_date'] = curr_date
        utils.conn_db('asset_wih').insert_one(item)

    @property
    def wih_record_fnv_hash(self):
        if False:
            print('Hello World!')
        if self._wih_record_fnv_hash is None:
            self._wih_record_fnv_hash = asset_wih.get_wih_record_fnv_hash(self.scope_id)
        return self._wih_record_fnv_hash

    def run(self):
        if False:
            return 10
        results = []
        self.init_scope_data()
        logger.info('run AssetWihMonitor, scope_id: {} sites: {}'.format(self.scope_id, len(self.sites)))
        if len(self.sites) == 0:
            return results
        wih_results = run_wih(self.sites)
        fnv_hash_set = set(self.wih_record_fnv_hash)
        for item in wih_results:
            item_fnv_hash = str(item.fnv_hash)
            if item_fnv_hash in fnv_hash_set:
                continue
            if item.recordType == 'domain':
                if self.scope_domains:
                    if not domain_in_scope_domain(item.content, self.scope_domains):
                        continue
                if check_domain_black(item.content):
                    continue
            self.save_asset_wih_record(item)
            results.append(item)
            fnv_hash_set.add(item_fnv_hash)
        logger.info('AssetWihMonitor, scope_id: {} results: {}'.format(self.scope_id, len(results)))
        self._wih_record_fnv_hash = None
        return results

def asset_wih_monitor(scope_id: str):
    if False:
        print('Hello World!')
    monitor = AssetWihMonitor(scope_id)
    results = monitor.run()
    return results

def domain_in_scope_domain(domain: str, scope_domain: list):
    if False:
        i = 10
        return i + 15
    for scope in scope_domain:
        if domain.endswith('.' + scope):
            return True
    return False