import hashlib
import json
from collections import defaultdict
from html import unescape
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from autoscraper.utils import FuzzyText, ResultItem, get_non_rec_text, get_random_str, normalize, text_match, unique_hashable, unique_stack_list

class AutoScraper(object):
    """
    AutoScraper : A Smart, Automatic, Fast and Lightweight Web Scraper for Python.
    AutoScraper automatically learns a set of rules required to extract the needed content
        from a web page. So the programmer doesn't need to explicitly construct the rules.

    Attributes
    ----------
    stack_list: list
        List of rules learned by AutoScraper

    Methods
    -------
    build() - Learns a set of rules represented as stack_list based on the wanted_list,
        which can be reused for scraping similar elements from other web pages in the future.
    get_result_similar() - Gets similar results based on the previously learned rules.
    get_result_exact() - Gets exact results based on the previously learned rules.
    get_results() - Gets exact and similar results based on the previously learned rules.
    save() - Serializes the stack_list as JSON and saves it to disk.
    load() - De-serializes the JSON representation of the stack_list and loads it back.
    remove_rules() - Removes one or more learned rule[s] from the stack_list.
    keep_rules() - Keeps only the specified learned rules in the stack_list and removes the others.
    """
    request_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36             (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}

    def __init__(self, stack_list=None):
        if False:
            i = 10
            return i + 15
        self.stack_list = stack_list or []

    def save(self, file_path):
        if False:
            print('Hello World!')
        '\n        Serializes the stack_list as JSON and saves it to the disk.\n\n        Parameters\n        ----------\n        file_path: str\n            Path of the JSON output\n\n        Returns\n        -------\n        None\n        '
        data = dict(stack_list=self.stack_list)
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def load(self, file_path):
        if False:
            while True:
                i = 10
        '\n        De-serializes the JSON representation of the stack_list and loads it back.\n\n        Parameters\n        ----------\n        file_path: str\n            Path of the JSON file to load stack_list from.\n\n        Returns\n        -------\n        None\n        '
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            self.stack_list = data
            return
        self.stack_list = data['stack_list']

    @classmethod
    def _fetch_html(cls, url, request_args=None):
        if False:
            return 10
        request_args = request_args or {}
        headers = dict(cls.request_headers)
        if url:
            headers['Host'] = urlparse(url).netloc
        user_headers = request_args.pop('headers', {})
        headers.update(user_headers)
        res = requests.get(url, headers=headers, **request_args)
        if res.encoding == 'ISO-8859-1' and (not 'ISO-8859-1' in res.headers.get('Content-Type', '')):
            res.encoding = res.apparent_encoding
        html = res.text
        return html

    @classmethod
    def _get_soup(cls, url=None, html=None, request_args=None):
        if False:
            print('Hello World!')
        if html:
            html = normalize(unescape(html))
            return BeautifulSoup(html, 'lxml')
        html = cls._fetch_html(url, request_args)
        html = normalize(unescape(html))
        return BeautifulSoup(html, 'lxml')

    @staticmethod
    def _get_valid_attrs(item):
        if False:
            return 10
        key_attrs = {'class', 'style'}
        attrs = {k: v if v != [] else '' for (k, v) in item.attrs.items() if k in key_attrs}
        for attr in key_attrs:
            if attr not in attrs:
                attrs[attr] = ''
        return attrs

    @staticmethod
    def _child_has_text(child, text, url, text_fuzz_ratio):
        if False:
            print('Hello World!')
        child_text = child.getText().strip()
        if text_match(text, child_text, text_fuzz_ratio):
            parent_text = child.parent.getText().strip()
            if child_text == parent_text and child.parent.parent:
                return False
            child.wanted_attr = None
            return True
        if text_match(text, get_non_rec_text(child), text_fuzz_ratio):
            child.is_non_rec_text = True
            child.wanted_attr = None
            return True
        for (key, value) in child.attrs.items():
            if not isinstance(value, str):
                continue
            value = value.strip()
            if text_match(text, value, text_fuzz_ratio):
                child.wanted_attr = key
                return True
            if key in {'href', 'src'}:
                full_url = urljoin(url, value)
                if text_match(text, full_url, text_fuzz_ratio):
                    child.wanted_attr = key
                    child.is_full_url = True
                    return True
        return False

    def _get_children(self, soup, text, url, text_fuzz_ratio):
        if False:
            for i in range(10):
                print('nop')
        children = reversed(soup.findChildren())
        children = [x for x in children if self._child_has_text(x, text, url, text_fuzz_ratio)]
        return children

    def build(self, url=None, wanted_list=None, wanted_dict=None, html=None, request_args=None, update=False, text_fuzz_ratio=1.0):
        if False:
            return 10
        '\n        Automatically constructs a set of rules to scrape the specified target[s] from a web page.\n            The rules are represented as stack_list.\n\n        Parameters:\n        ----------\n        url: str, optional\n            URL of the target web page. You should either pass url or html or both.\n\n        wanted_list: list of strings or compiled regular expressions, optional\n            A list of needed contents to be scraped.\n                AutoScraper learns a set of rules to scrape these targets. If specified,\n                wanted_dict will be ignored.\n\n        wanted_dict: dict, optional\n            A dict of needed contents to be scraped. Keys are aliases and values are list of target texts\n                or compiled regular expressions.\n                AutoScraper learns a set of rules to scrape these targets and sets its aliases.\n\n        html: str, optional\n            An HTML string can also be passed instead of URL.\n                You should either pass url or html or both.\n\n        request_args: dict, optional\n            A dictionary used to specify a set of additional request parameters used by requests\n                module. You can specify proxy URLs, custom headers etc.\n\n        update: bool, optional, defaults to False\n            If True, new learned rules will be added to the previous ones.\n            If False, all previously learned rules will be removed.\n\n        text_fuzz_ratio: float in range [0, 1], optional, defaults to 1.0\n            The fuzziness ratio threshold for matching the wanted contents.\n\n        Returns:\n        --------\n        List of similar results\n        '
        soup = self._get_soup(url=url, html=html, request_args=request_args)
        result_list = []
        if update is False:
            self.stack_list = []
        if wanted_list:
            wanted_dict = {'': wanted_list}
        wanted_list = []
        for (alias, wanted_items) in wanted_dict.items():
            wanted_items = [normalize(w) for w in wanted_items]
            wanted_list += wanted_items
            for wanted in wanted_items:
                children = self._get_children(soup, wanted, url, text_fuzz_ratio)
                for child in children:
                    (result, stack) = self._get_result_for_child(child, soup, url)
                    stack['alias'] = alias
                    result_list += result
                    self.stack_list.append(stack)
        result_list = [item.text for item in result_list]
        result_list = unique_hashable(result_list)
        self.stack_list = unique_stack_list(self.stack_list)
        return result_list

    @classmethod
    def _build_stack(cls, child, url):
        if False:
            for i in range(10):
                print('nop')
        content = [(child.name, cls._get_valid_attrs(child))]
        parent = child
        while True:
            grand_parent = parent.findParent()
            if not grand_parent:
                break
            children = grand_parent.findAll(parent.name, cls._get_valid_attrs(parent), recursive=False)
            for (i, c) in enumerate(children):
                if c == parent:
                    content.insert(0, (grand_parent.name, cls._get_valid_attrs(grand_parent), i))
                    break
            if not grand_parent.parent:
                break
            parent = grand_parent
        wanted_attr = getattr(child, 'wanted_attr', None)
        is_full_url = getattr(child, 'is_full_url', False)
        is_non_rec_text = getattr(child, 'is_non_rec_text', False)
        stack = dict(content=content, wanted_attr=wanted_attr, is_full_url=is_full_url, is_non_rec_text=is_non_rec_text)
        stack['url'] = url if is_full_url else ''
        stack['hash'] = hashlib.sha256(str(stack).encode('utf-8')).hexdigest()
        stack['stack_id'] = 'rule_' + get_random_str(4)
        return stack

    def _get_result_for_child(self, child, soup, url):
        if False:
            print('Hello World!')
        stack = self._build_stack(child, url)
        result = self._get_result_with_stack(stack, soup, url, 1.0)
        return (result, stack)

    @staticmethod
    def _fetch_result_from_child(child, wanted_attr, is_full_url, url, is_non_rec_text):
        if False:
            i = 10
            return i + 15
        if wanted_attr is None:
            if is_non_rec_text:
                return get_non_rec_text(child)
            return child.getText().strip()
        if wanted_attr not in child.attrs:
            return None
        if is_full_url:
            return urljoin(url, child.attrs[wanted_attr])
        return child.attrs[wanted_attr]

    @staticmethod
    def _get_fuzzy_attrs(attrs, attr_fuzz_ratio):
        if False:
            while True:
                i = 10
        attrs = dict(attrs)
        for (key, val) in attrs.items():
            if isinstance(val, str) and val:
                val = FuzzyText(val, attr_fuzz_ratio)
            elif isinstance(val, (list, tuple)):
                val = [FuzzyText(x, attr_fuzz_ratio) if x else x for x in val]
            attrs[key] = val
        return attrs

    def _get_result_with_stack(self, stack, soup, url, attr_fuzz_ratio, **kwargs):
        if False:
            print('Hello World!')
        parents = [soup]
        stack_content = stack['content']
        contain_sibling_leaves = kwargs.get('contain_sibling_leaves', False)
        for (index, item) in enumerate(stack_content):
            children = []
            if item[0] == '[document]':
                continue
            for parent in parents:
                attrs = item[1]
                if attr_fuzz_ratio < 1.0:
                    attrs = self._get_fuzzy_attrs(attrs, attr_fuzz_ratio)
                found = parent.findAll(item[0], attrs, recursive=False)
                if not found:
                    continue
                if not contain_sibling_leaves and index == len(stack_content) - 1:
                    idx = min(len(found) - 1, stack_content[index - 1][2])
                    found = [found[idx]]
                children += found
            parents = children
        wanted_attr = stack['wanted_attr']
        is_full_url = stack['is_full_url']
        is_non_rec_text = stack.get('is_non_rec_text', False)
        result = [ResultItem(self._fetch_result_from_child(i, wanted_attr, is_full_url, url, is_non_rec_text), getattr(i, 'child_index', 0)) for i in parents]
        if not kwargs.get('keep_blank', False):
            result = [x for x in result if x.text]
        return result

    def _get_result_with_stack_index_based(self, stack, soup, url, attr_fuzz_ratio, **kwargs):
        if False:
            print('Hello World!')
        p = soup.findChildren(recursive=False)[0]
        stack_content = stack['content']
        for (index, item) in enumerate(stack_content[:-1]):
            if item[0] == '[document]':
                continue
            content = stack_content[index + 1]
            attrs = content[1]
            if attr_fuzz_ratio < 1.0:
                attrs = self._get_fuzzy_attrs(attrs, attr_fuzz_ratio)
            p = p.findAll(content[0], attrs, recursive=False)
            if not p:
                return []
            idx = min(len(p) - 1, item[2])
            p = p[idx]
        result = [ResultItem(self._fetch_result_from_child(p, stack['wanted_attr'], stack['is_full_url'], url, stack['is_non_rec_text']), getattr(p, 'child_index', 0))]
        if not kwargs.get('keep_blank', False):
            result = [x for x in result if x.text]
        return result

    def _get_result_by_func(self, func, url, html, soup, request_args, grouped, group_by_alias, unique, attr_fuzz_ratio, **kwargs):
        if False:
            while True:
                i = 10
        if not soup:
            soup = self._get_soup(url=url, html=html, request_args=request_args)
        keep_order = kwargs.get('keep_order', False)
        if group_by_alias or (keep_order and (not grouped)):
            for (index, child) in enumerate(soup.findChildren()):
                setattr(child, 'child_index', index)
        result_list = []
        grouped_result = defaultdict(list)
        for stack in self.stack_list:
            if not url:
                url = stack.get('url', '')
            result = func(stack, soup, url, attr_fuzz_ratio, **kwargs)
            if not grouped and (not group_by_alias):
                result_list += result
                continue
            group_id = stack.get('alias', '') if group_by_alias else stack['stack_id']
            grouped_result[group_id] += result
        return self._clean_result(result_list, grouped_result, grouped, group_by_alias, unique, keep_order)

    @staticmethod
    def _clean_result(result_list, grouped_result, grouped, grouped_by_alias, unique, keep_order):
        if False:
            i = 10
            return i + 15
        if not grouped and (not grouped_by_alias):
            if unique is None:
                unique = True
            if keep_order:
                result_list = sorted(result_list, key=lambda x: x.index)
            result = [x.text for x in result_list]
            if unique:
                result = unique_hashable(result)
            return result
        for (k, val) in grouped_result.items():
            if grouped_by_alias:
                val = sorted(val, key=lambda x: x.index)
            val = [x.text for x in val]
            if unique:
                val = unique_hashable(val)
            grouped_result[k] = val
        return dict(grouped_result)

    def get_result_similar(self, url=None, html=None, soup=None, request_args=None, grouped=False, group_by_alias=False, unique=None, attr_fuzz_ratio=1.0, keep_blank=False, keep_order=False, contain_sibling_leaves=False):
        if False:
            while True:
                i = 10
        '\n        Gets similar results based on the previously learned rules.\n\n        Parameters:\n        ----------\n        url: str, optional\n            URL of the target web page. You should either pass url or html or both.\n\n        html: str, optional\n            An HTML string can also be passed instead of URL.\n                You should either pass url or html or both.\n\n        request_args: dict, optional\n            A dictionary used to specify a set of additional request parameters used by requests\n                module. You can specify proxy URLs, custom headers etc.\n\n        grouped: bool, optional, defaults to False\n            If set to True, the result will be a dictionary with the rule_ids as keys\n                and a list of scraped data per rule as values.\n\n        group_by_alias: bool, optional, defaults to False\n            If set to True, the result will be a dictionary with the rule alias as keys\n                and a list of scraped data per alias as values.\n\n        unique: bool, optional, defaults to True for non grouped results and\n                False for grouped results.\n            If set to True, will remove duplicates from returned result list.\n\n        attr_fuzz_ratio: float in range [0, 1], optional, defaults to 1.0\n            The fuzziness ratio threshold for matching html tag attributes.\n\n        keep_blank: bool, optional, defaults to False\n            If set to True, missing values will be returned as empty strings.\n\n        keep_order: bool, optional, defaults to False\n            If set to True, the results will be ordered as they are present on the web page.\n\n        contain_sibling_leaves: bool, optional, defaults to False\n            If set to True, the results will also contain the sibling leaves of the wanted elements.\n\n        Returns:\n        --------\n        List of similar results scraped from the web page.\n        Dictionary if grouped=True or group_by_alias=True.\n        '
        func = self._get_result_with_stack
        return self._get_result_by_func(func, url, html, soup, request_args, grouped, group_by_alias, unique, attr_fuzz_ratio, keep_blank=keep_blank, keep_order=keep_order, contain_sibling_leaves=contain_sibling_leaves)

    def get_result_exact(self, url=None, html=None, soup=None, request_args=None, grouped=False, group_by_alias=False, unique=None, attr_fuzz_ratio=1.0, keep_blank=False):
        if False:
            return 10
        '\n        Gets exact results based on the previously learned rules.\n\n        Parameters:\n        ----------\n        url: str, optional\n            URL of the target web page. You should either pass url or html or both.\n\n        html: str, optional\n            An HTML string can also be passed instead of URL.\n                You should either pass url or html or both.\n\n        request_args: dict, optional\n            A dictionary used to specify a set of additional request parameters used by requests\n                module. You can specify proxy URLs, custom headers etc.\n\n        grouped: bool, optional, defaults to False\n            If set to True, the result will be a dictionary with the rule_ids as keys\n                and a list of scraped data per rule as values.\n\n        group_by_alias: bool, optional, defaults to False\n            If set to True, the result will be a dictionary with the rule alias as keys\n                and a list of scraped data per alias as values.\n\n        unique: bool, optional, defaults to True for non grouped results and\n                False for grouped results.\n            If set to True, will remove duplicates from returned result list.\n\n        attr_fuzz_ratio: float in range [0, 1], optional, defaults to 1.0\n            The fuzziness ratio threshold for matching html tag attributes.\n\n        keep_blank: bool, optional, defaults to False\n            If set to True, missing values will be returned as empty strings.\n\n        Returns:\n        --------\n        List of exact results scraped from the web page.\n        Dictionary if grouped=True or group_by_alias=True.\n        '
        func = self._get_result_with_stack_index_based
        return self._get_result_by_func(func, url, html, soup, request_args, grouped, group_by_alias, unique, attr_fuzz_ratio, keep_blank=keep_blank)

    def get_result(self, url=None, html=None, request_args=None, grouped=False, group_by_alias=False, unique=None, attr_fuzz_ratio=1.0):
        if False:
            i = 10
            return i + 15
        '\n        Gets similar and exact results based on the previously learned rules.\n\n        Parameters:\n        ----------\n        url: str, optional\n            URL of the target web page. You should either pass url or html or both.\n\n        html: str, optional\n            An HTML string can also be passed instead of URL.\n                You should either pass url or html or both.\n\n        request_args: dict, optional\n            A dictionary used to specify a set of additional request parameters used by requests\n                module. You can specify proxy URLs, custom headers etc.\n\n        grouped: bool, optional, defaults to False\n            If set to True, the result will be dictionaries with the rule_ids as keys\n                and a list of scraped data per rule as values.\n\n        group_by_alias: bool, optional, defaults to False\n            If set to True, the result will be a dictionary with the rule alias as keys\n                and a list of scraped data per alias as values.\n\n        unique: bool, optional, defaults to True for non grouped results and\n                False for grouped results.\n            If set to True, will remove duplicates from returned result list.\n\n        attr_fuzz_ratio: float in range [0, 1], optional, defaults to 1.0\n            The fuzziness ratio threshold for matching html tag attributes.\n\n        Returns:\n        --------\n        Pair of (similar, exact) results.\n        See get_result_similar and get_result_exact methods.\n        '
        soup = self._get_soup(url=url, html=html, request_args=request_args)
        args = dict(url=url, soup=soup, grouped=grouped, group_by_alias=group_by_alias, unique=unique, attr_fuzz_ratio=attr_fuzz_ratio)
        similar = self.get_result_similar(**args)
        exact = self.get_result_exact(**args)
        return (similar, exact)

    def remove_rules(self, rules):
        if False:
            return 10
        '\n        Removes a list of learned rules from stack_list.\n\n        Parameters:\n        ----------\n        rules : list\n            A list of rules to be removed\n\n        Returns:\n        --------\n        None\n        '
        self.stack_list = [x for x in self.stack_list if x['stack_id'] not in rules]

    def keep_rules(self, rules):
        if False:
            i = 10
            return i + 15
        '\n        Removes all other rules except the specified ones.\n\n        Parameters:\n        ----------\n        rules : list\n            A list of rules to keep in stack_list and removing the rest.\n\n        Returns:\n        --------\n        None\n        '
        self.stack_list = [x for x in self.stack_list if x['stack_id'] in rules]

    def set_rule_aliases(self, rule_aliases):
        if False:
            i = 10
            return i + 15
        '\n        Sets the specified alias for each rule\n\n        Parameters:\n        ----------\n        rule_aliases : dict\n            A dictionary with keys of rule_id and values of alias\n\n        Returns:\n        --------\n        None\n        '
        id_to_stack = {stack['stack_id']: stack for stack in self.stack_list}
        for (rule_id, alias) in rule_aliases.items():
            id_to_stack[rule_id]['alias'] = alias

    def generate_python_code(self):
        if False:
            i = 10
            return i + 15
        print('This function is deprecated. Please use save() and load() instead.')