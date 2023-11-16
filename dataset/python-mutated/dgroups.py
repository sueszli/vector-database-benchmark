import collections
import libqtile.hook
from libqtile.backend.base import Static
from libqtile.command import lazy
from libqtile.config import Group, Key, Match, Rule
from libqtile.log_utils import logger

def simple_key_binder(mod, keynames=None):
    if False:
        i = 10
        return i + 15
    'Bind keys to mod+group position or to the keys specified as second argument'

    def func(dgroup):
        if False:
            i = 10
            return i + 15
        for key in dgroup.keys[:]:
            dgroup.qtile.ungrab_key(key)
            dgroup.qtile.config.keys.remove(key)
            dgroup.keys.remove(key)
        if keynames:
            keys = keynames
        else:
            keys = list(map(str, list(range(1, 10)) + [0]))
        for (keyname, group) in zip(keys, dgroup.qtile.groups):
            name = group.name
            key = Key([mod], keyname, lazy.group[name].toscreen())
            key_s = Key([mod, 'shift'], keyname, lazy.window.togroup(name))
            key_c = Key([mod, 'control'], keyname, lazy.group.switch_groups(name))
            dgroup.keys.extend([key, key_s, key_c])
            dgroup.qtile.config.keys.extend([key, key_s, key_c])
            dgroup.qtile.grab_key(key)
            dgroup.qtile.grab_key(key_s)
            dgroup.qtile.grab_key(key_c)
    return func

class DGroups:
    """Dynamic Groups"""

    def __init__(self, qtile, dgroups, key_binder=None, delay=1):
        if False:
            i = 10
            return i + 15
        self.qtile = qtile
        self.groups = dgroups
        self.groups_map = {}
        self.rules = []
        self.rules_map = {}
        self.last_rule_id = 0
        for rule in getattr(qtile.config, 'dgroups_app_rules', []):
            self.add_rule(rule)
        self.keys = []
        self.key_binder = key_binder
        self._setup_hooks()
        self._setup_groups()
        self.delay = delay
        self.timeout = {}

    def add_rule(self, rule, last=True):
        if False:
            print('Hello World!')
        rule_id = self.last_rule_id
        self.rules_map[rule_id] = rule
        if last:
            self.rules.append(rule)
        else:
            self.rules.insert(0, rule)
        self.last_rule_id += 1
        return rule_id

    def remove_rule(self, rule_id):
        if False:
            while True:
                i = 10
        rule = self.rules_map.get(rule_id)
        if rule:
            self.rules.remove(rule)
            del self.rules_map[rule_id]
        else:
            logger.warning('Rule "%s" not found', rule_id)

    def add_dgroup(self, group, start=False):
        if False:
            for i in range(10):
                print('nop')
        self.groups_map[group.name] = group
        rule = Rule(group.matches, group=group.name)
        self.rules.append(rule)
        if start:
            self.qtile.add_group(group.name, group.layout, group.layouts, group.label, screen_affinity=group.screen_affinity)

    def _setup_groups(self):
        if False:
            return 10
        for group in self.groups:
            self.add_dgroup(group, group.init)
            if group.spawn and (not self.qtile.no_spawn):
                if isinstance(group.spawn, str):
                    spawns = [group.spawn]
                else:
                    spawns = group.spawn
                for spawn in spawns:
                    pid = self.qtile.spawn(spawn)
                    self.add_rule(Rule(Match(net_wm_pid=pid), group.name))

    def _setup_hooks(self):
        if False:
            i = 10
            return i + 15
        libqtile.hook.subscribe.addgroup(self._addgroup)
        libqtile.hook.subscribe.client_new(self._add)
        libqtile.hook.subscribe.client_killed(self._del)
        if self.key_binder:
            libqtile.hook.subscribe.setgroup(lambda : self.key_binder(self))
            libqtile.hook.subscribe.changegroup(lambda : self.key_binder(self))

    def _addgroup(self, group_name):
        if False:
            i = 10
            return i + 15
        if group_name not in self.groups_map:
            self.add_dgroup(Group(group_name, persist=False))

    def _add(self, client):
        if False:
            return 10
        if client in self.timeout:
            logger.debug('Remove dgroup source')
            self.timeout.pop(client).cancel()
        if isinstance(client, Static):
            return
        if client.group is not None:
            return
        group_set = False
        intrusive = False
        for rule in self.rules:
            if rule.matches(client):
                if rule.group:
                    if rule.group in self.groups_map:
                        layout = self.groups_map[rule.group].layout
                        layouts = self.groups_map[rule.group].layouts
                        label = self.groups_map[rule.group].label
                    else:
                        layout = None
                        layouts = None
                        label = None
                    group_added = self.qtile.add_group(rule.group, layout, layouts, label)
                    client.togroup(rule.group)
                    group_set = True
                    group_obj = self.qtile.groups_map[rule.group]
                    group = self.groups_map.get(rule.group)
                    if group and group_added:
                        for (k, v) in list(group.layout_opts.items()):
                            if isinstance(v, collections.abc.Callable):
                                v(group_obj.layout)
                            else:
                                setattr(group_obj.layout, k, v)
                        affinity = group.screen_affinity
                        if affinity and len(self.qtile.screens) > affinity:
                            self.qtile.screens[affinity].set_group(group_obj)
                if rule.float:
                    client.enable_floating()
                if rule.intrusive:
                    intrusive = rule.intrusive
                if rule.break_on_match:
                    break
        if not group_set:
            current_group = self.qtile.current_group.name
            if current_group in self.groups_map and self.groups_map[current_group].exclusive and (not intrusive):
                wm_class = client.get_wm_class()
                if wm_class:
                    if len(wm_class) > 1:
                        wm_class = wm_class[1]
                    else:
                        wm_class = wm_class[0]
                    group_name = wm_class
                else:
                    group_name = client.name or 'Unnamed'
                self.add_dgroup(Group(group_name, persist=False), start=True)
                client.togroup(group_name)
        self.sort_groups()

    def sort_groups(self):
        if False:
            print('Hello World!')
        grps = self.qtile.groups
        sorted_grps = sorted(grps, key=lambda g: self.groups_map[g.name].position)
        if grps != sorted_grps:
            self.qtile.groups = sorted_grps
            libqtile.hook.fire('changegroup')

    def _del(self, client):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(client, Static):
            return
        group = client.group

        def delete_client():
            if False:
                return 10
            if group and group.name in self.groups_map and (not self.groups_map[group.name].persist) and (len(group.windows) <= 0):
                self.qtile.delete_group(group.name)
                self.sort_groups()
            del self.timeout[client]
        logger.debug('Deleting %s in %ss', group, self.delay)
        self.timeout[client] = self.qtile.call_later(self.delay, delete_client)