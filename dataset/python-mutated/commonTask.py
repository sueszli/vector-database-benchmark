import time
from urllib.parse import urlparse
from bson import ObjectId
from app import utils
from app import services
from app.config import Config
from app.modules import CollectSource, WebSiteFetchStatus, WebSiteFetchOption
from app.services.nuclei_scan import nuclei_scan
from app.services import run_risk_cruising, BaseUpdateTask
logger = utils.get_logger()

class CommonTask(object):

    def __init__(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        self.task_id = task_id

    def insert_task_stat(self):
        if False:
            return 10
        query = {'_id': ObjectId(self.task_id)}
        stat = utils.arl.task_statistic(self.task_id)
        logger.info('insert task stat')
        update = {'$set': {'statistic': stat}}
        utils.conn_db('task').update_one(query, update)

    def insert_finger_stat(self):
        if False:
            return 10
        finger_stat_map = utils.arl.gen_stat_finger_map(self.task_id)
        logger.info('insert finger stat {}'.format(len(finger_stat_map)))
        for key in finger_stat_map:
            data = finger_stat_map[key].copy()
            data['task_id'] = self.task_id
            utils.conn_db('stat_finger').insert_one(data)

    def insert_cip_stat(self):
        if False:
            for i in range(10):
                print('nop')
        cip_map = utils.arl.gen_cip_map(self.task_id)
        logger.info('insert cip stat {}'.format(len(cip_map)))
        for cidr_ip in cip_map:
            item = cip_map[cidr_ip]
            ip_list = list(item['ip_set'])
            domain_list = list(item['domain_set'])
            data = {'cidr_ip': cidr_ip, 'ip_count': len(ip_list), 'ip_list': ip_list, 'domain_count': len(domain_list), 'domain_list': domain_list, 'task_id': self.task_id}
            utils.conn_db('cip').insert_one(data)

    def sync_asset(self):
        if False:
            i = 10
            return i + 15
        options = getattr(self, 'options', {})
        if not options:
            logger.warning('not found options {}'.format(self.task_id))
            return
        related_scope_id = options.get('related_scope_id', '')
        if not related_scope_id:
            return
        if len(related_scope_id) != 24:
            logger.warning('related_scope_id len not eq 24 {}'.format(self.task_id, related_scope_id))
            return
        services.sync_asset(task_id=self.task_id, scope_id=related_scope_id)

    def common_run(self):
        if False:
            return 10
        self.insert_finger_stat()
        self.insert_cip_stat()
        self.insert_task_stat()
        self.sync_asset()

class WebSiteFetch(object):

    def __init__(self, task_id: str, sites: list, options: dict, scope_domain: list=None):
        if False:
            return 10
        self.task_id = task_id
        self.sites = sites
        self.options = options
        self.base_update_task = BaseUpdateTask(self.task_id)
        self.site_info_list = []
        self.available_sites = []
        self.web_analyze_map = dict()
        self.wih_domain_set = set()
        self.wih_record_set = set()
        if not scope_domain:
            scope_domain = []
        self.scope_domain = scope_domain
        self.page_url_set = set()
        self.search_engines_result = dict()
        self._poc_sites = None
        self._task_domain_set = None

    @property
    def task_domain_set(self):
        if False:
            print('Hello World!')
        if self._task_domain_set is None:
            self._task_domain_set = set(utils.arl.get_domain_by_id(self.task_id))
        return self._task_domain_set

    def site_identify(self):
        if False:
            i = 10
            return i + 15
        self.web_analyze_map = services.web_analyze(self.available_sites)

    def __str__(self):
        if False:
            return 10
        return '<WebSiteFetch> task_id:{}, sites: {}, available_sites:{}'.format(self.task_id, len(self.sites), len(self.available_sites))

    def save_site_info(self):
        if False:
            for i in range(10):
                print('nop')
        for site_info in self.site_info_list:
            curr_site = site_info['site']
            site_path = '/image/' + self.task_id
            file_name = '{}/{}.jpg'.format(site_path, utils.gen_filename(curr_site))
            site_info['task_id'] = self.task_id
            site_info['screenshot'] = file_name
            if self.web_analyze_map:
                finger_list = self.web_analyze_map.get(curr_site, [])
                known_finger_set = set()
                for finger_item in site_info['finger']:
                    known_finger_set.add(finger_item['name'].lower())
                for analyze_finger in finger_list:
                    analyze_name = analyze_finger['name'].lower()
                    if analyze_name not in known_finger_set:
                        site_info['finger'].append(analyze_finger)
        logger.info('save_site_info site:{}, {}'.format(len(self.site_info_list), self.__str__()))
        if self.site_info_list:
            utils.conn_db('site').insert_many(self.site_info_list)

    def site_screenshot(self):
        if False:
            print('Hello World!')
        capture_save_dir = Config.SCREENSHOT_DIR + '/' + self.task_id
        services.site_screenshot(self.available_sites, concurrency=6, capture_dir=capture_save_dir)

    def site_spider(self):
        if False:
            print('Hello World!')
        entry_urls_list = []
        for site in self.available_sites:
            o = urlparse(site)
            if o.path != '':
                continue
            entry_urls = [site]
            entry_urls.extend(self.search_engines_result.get(site, []))
            entry_urls_list.append(entry_urls)
        site_spider_result = services.site_spider_thread(entry_urls_list)
        spider_urls = []
        for site in site_spider_result:
            target_urls = site_spider_result[site]
            new_target_urls = []
            for url in target_urls:
                if url in self.page_url_set:
                    continue
                new_target_urls.append(url)
                self.page_url_set.add(url)
            if not new_target_urls:
                continue
            spider_urls.extend(new_target_urls)
        if len(spider_urls) > 0:
            logger.info('spider_urls {} task_id:{}'.format(len(spider_urls), self.task_id))
            page_map = services.page_fetch(spider_urls)
            for url in page_map:
                item = build_url_item(url, self.task_id, source=CollectSource.SITESPIDER)
                item.update(page_map[url])
                utils.conn_db('url').insert_one(item)

    def fetch_site(self):
        if False:
            return 10
        self.site_info_list = services.fetch_site(self.sites)
        for site_info in self.site_info_list:
            curr_site = site_info['site']
            self.available_sites.append(curr_site)

    def file_leak(self):
        if False:
            while True:
                i = 10
        for site in self.poc_sites:
            pages = services.file_leak([site], utils.load_file(Config.FILE_LEAK_TOP_2k))
            for page in pages:
                item = page.dump_json()
                item['task_id'] = self.task_id
                item['site'] = site
                utils.conn_db('fileleak').insert_one(item)

    @property
    def poc_sites(self):
        if False:
            print('Hello World!')
        if self._poc_sites is None:
            self._poc_sites = set()
            for x in self.available_sites:
                cut_target = utils.url.cut_filename(x)
                if cut_target:
                    self._poc_sites.add(cut_target)
        return self._poc_sites

    def risk_cruising(self, npoc_service_target_set: set):
        if False:
            print('Hello World!')
        poc_config = self.options.get('poc_config', [])
        plugins = []
        for info in poc_config:
            if not info.get('enable'):
                continue
            plugins.append(info['plugin_name'])
        poc_targets = self.poc_sites
        if npoc_service_target_set is not None:
            poc_targets = self.poc_sites | npoc_service_target_set
        result = run_risk_cruising(plugins=plugins, targets=poc_targets)
        for item in result:
            item['task_id'] = self.task_id
            item['save_date'] = utils.curr_date()
            utils.conn_db('vuln').insert_one(item)

    def nuclei_scan(self):
        if False:
            while True:
                i = 10
        logger.info('start nuclei_scan， poc_sites:{}'.format(len(self.poc_sites)))
        scan_results = nuclei_scan(list(self.poc_sites))
        for item in scan_results:
            item['task_id'] = self.task_id
            item['save_date'] = utils.curr_date()
            utils.conn_db('nuclei_result').insert_one(item)
        logger.info('end nuclei_scan， result:{}'.format(len(scan_results)))

    def run_func(self, name: str, func: callable):
        if False:
            for i in range(10):
                print('nop')
        logger.info('start run {}, {}'.format(name, self.__str__()))
        self.base_update_task.update_task_field('status', name)
        t1 = time.time()
        func()
        elapse = time.time() - t1
        self.base_update_task.update_services(name, elapse)
        logger.info('end run {} ({:.2f}s), {}'.format(name, elapse, self.__str__()))

    def update_page_url_set(self):
        if False:
            return 10
        from app.helpers import get_url_by_task_id
        urls = get_url_by_task_id(self.task_id)
        self.page_url_set |= set(urls)
        for u in self.page_url_set:
            o = urlparse(u)
            ret_url = '{}://{}'.format(o.scheme, o.netloc)
            entry_urls = self.search_engines_result.get(ret_url, [])
            entry_urls.append(u)
            self.search_engines_result[ret_url] = entry_urls

    def add_wih_domain_set(self, record):
        if False:
            print('Hello World!')
        if self.scope_domain:
            if record.recordType == 'domain':
                if not domain_in_scope_domain(record.content, self.scope_domain):
                    return
                if utils.check_domain_black(record.content):
                    return
                if record.content in self.wih_domain_set:
                    return
                if record.content in self.wih_domain_set:
                    return
                self.wih_domain_set.add(record.content)

    def run_web_info_hunter(self):
        if False:
            return 10
        records = set(services.run_wih(self.sites))
        for record in records:
            if record.fnv_hash in self.wih_record_set:
                continue
            self.add_wih_domain_set(record)
            item = record.dump_json()
            item['task_id'] = self.task_id
            utils.conn_db('wih').insert_one(item)
            self.wih_record_set.add(record.fnv_hash)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_func(WebSiteFetchStatus.FETCH_SITE, self.fetch_site)
        ' *** 执行站点识别 '
        if self.options.get(WebSiteFetchOption.SITE_IDENTIFY):
            self.run_func(WebSiteFetchStatus.SITE_IDENTIFY, self.site_identify)
        ' *** 保存站点信息到数据库 '
        self.save_site_info()
        self.site_info_list = []
        ' *** 站点截图 '
        if self.options.get(WebSiteFetchOption.SITE_CAPTURE):
            self.run_func(WebSiteFetchStatus.SITE_CAPTURE, self.site_screenshot)
        ' ***调用站点爬虫发现URL '
        if self.options.get(WebSiteFetchOption.SITE_SPIDER):
            self.update_page_url_set()
            self.run_func(WebSiteFetchStatus.SITE_SPIDER, self.site_spider)
        ' *** 对站点进行文件目录爆破 '
        if self.options.get(WebSiteFetchOption.FILE_LEAK):
            self.run_func(WebSiteFetchStatus.FILE_LEAK, self.file_leak)
        ' *** 对站点运行 nuclei '
        if self.options.get(WebSiteFetchOption.NUCLEI_SCAN):
            self.run_func(WebSiteFetchStatus.NUCLEI_SCAN, self.nuclei_scan)
        ' *** 对站点调用 WebInfoHunter '
        if self.options.get(WebSiteFetchOption.Info_Hunter):
            self.run_func(WebSiteFetchStatus.Info_Hunter, self.run_web_info_hunter)

def domain_in_scope_domain(domain: str, scope_domain: list):
    if False:
        while True:
            i = 10
    for scope in scope_domain:
        if domain.endswith('.' + scope):
            return True
    return False

def build_url_item(site, task_id, source):
    if False:
        for i in range(10):
            print('nop')
    item = {'site': site, 'task_id': task_id, 'source': source}
    domain_parsed = utils.domain_parsed(site)
    if domain_parsed:
        item['fld'] = domain_parsed['fld']
    return item