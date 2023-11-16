""" Modules dependency graph. """
import os, sys, imp
from os.path import join as opj
import itertools
import zipimport
import odoo
import odoo.osv as osv
import odoo.tools as tools
import odoo.tools.osutil as osutil
from odoo.tools.translate import _
import zipfile
import odoo.release as release
import re
import base64
from zipfile import PyZipFile, ZIP_DEFLATED
from cStringIO import StringIO
import logging
_logger = logging.getLogger(__name__)

class Graph(dict):
    """ Modules dependency graph.

    The graph is a mapping from module name to Nodes.

    """

    def add_node(self, name, info):
        if False:
            i = 10
            return i + 15
        (max_depth, father) = (0, None)
        for d in info['depends']:
            n = self.get(d) or Node(d, self, None)
            if n.depth >= max_depth:
                father = n
                max_depth = n.depth
        if father:
            return father.add_child(name, info)
        else:
            return Node(name, self, info)

    def update_from_db(self, cr):
        if False:
            print('Hello World!')
        if not len(self):
            return
        additional_data = dict(((key, {'id': 0, 'state': 'uninstalled', 'dbdemo': False, 'installed_version': None}) for key in self.keys()))
        cr.execute('SELECT name, id, state, demo AS dbdemo, latest_version AS installed_version  FROM ir_module_module WHERE name IN %s', (tuple(additional_data),))
        additional_data.update(((x['name'], x) for x in cr.dictfetchall()))
        for package in self.values():
            for (k, v) in additional_data[package.name].items():
                setattr(package, k, v)

    def add_module(self, cr, module, force=None):
        if False:
            return 10
        self.add_modules(cr, [module], force)

    def add_modules(self, cr, module_list, force=None):
        if False:
            while True:
                i = 10
        if force is None:
            force = []
        packages = []
        len_graph = len(self)
        for module in module_list:
            info = odoo.modules.module.load_information_from_description_file(module)
            if info and info['installable']:
                packages.append((module, info))
            elif module != 'studio_customization':
                _logger.warning('module %s: not installable, skipped', module)
        dependencies = dict([(p, info['depends']) for (p, info) in packages])
        (current, later) = (set([p for (p, info) in packages]), set())
        while packages and current > later:
            (package, info) = packages[0]
            deps = info['depends']
            if reduce(lambda x, y: x and y in self, deps, True):
                if not package in current:
                    packages.pop(0)
                    continue
                later.clear()
                current.remove(package)
                node = self.add_node(package, info)
                for kind in ('init', 'demo', 'update'):
                    if package in tools.config[kind] or 'all' in tools.config[kind] or kind in force:
                        setattr(node, kind, True)
            else:
                later.add(package)
                packages.append((package, info))
            packages.pop(0)
        self.update_from_db(cr)
        for package in later:
            unmet_deps = filter(lambda p: p not in self, dependencies[package])
            _logger.error('module %s: Unmet dependencies: %s', package, ', '.join(unmet_deps))
        return len(self) - len_graph

    def __iter__(self):
        if False:
            return 10
        level = 0
        done = set(self.keys())
        while done:
            level_modules = sorted(((name, module) for (name, module) in self.items() if module.depth == level))
            for (name, module) in level_modules:
                done.remove(name)
                yield module
            level += 1

    def __str__(self):
        if False:
            print('Hello World!')
        return '\n'.join((str(n) for n in self if n.depth == 0))

class Node(object):
    """ One module in the modules dependency graph.

    Node acts as a per-module singleton. A node is constructed via
    Graph.add_module() or Graph.add_modules(). Some of its fields are from
    ir_module_module (setted by Graph.update_from_db()).

    """

    def __new__(cls, name, graph, info):
        if False:
            print('Hello World!')
        if name in graph:
            inst = graph[name]
        else:
            inst = object.__new__(cls)
            graph[name] = inst
        return inst

    def __init__(self, name, graph, info):
        if False:
            while True:
                i = 10
        self.name = name
        self.graph = graph
        self.info = info or getattr(self, 'info', {})
        if not hasattr(self, 'children'):
            self.children = []
        if not hasattr(self, 'depth'):
            self.depth = 0

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        return self.info

    def add_child(self, name, info):
        if False:
            for i in range(10):
                print('nop')
        node = Node(name, self.graph, info)
        node.depth = self.depth + 1
        if node not in self.children:
            self.children.append(node)
        for attr in ('init', 'update', 'demo'):
            if hasattr(self, attr):
                setattr(node, attr, True)
        self.children.sort(lambda x, y: cmp(x.name, y.name))
        return node

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        super(Node, self).__setattr__(name, value)
        if name in ('init', 'update', 'demo'):
            tools.config[name][self.name] = 1
            for child in self.children:
                setattr(child, name, value)
        if name == 'depth':
            for child in self.children:
                setattr(child, name, value + 1)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return itertools.chain(iter(self.children), *map(iter, self.children))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._pprint()

    def _pprint(self, depth=0):
        if False:
            for i in range(10):
                print('nop')
        s = '%s\n' % self.name
        for c in self.children:
            s += '%s`-> %s' % ('   ' * depth, c._pprint(depth + 1))
        return s