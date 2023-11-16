from sacred.utils import iter_prefixes, join_paths

class ConfigSummary(dict):

    def __init__(self, added=(), modified=(), typechanged=(), ignored_fallbacks=(), docs=()):
        if False:
            print('Hello World!')
        super().__init__()
        self.added = set(added)
        self.modified = set(modified)
        self.typechanged = dict(typechanged)
        self.ignored_fallbacks = set(ignored_fallbacks)
        self.docs = dict(docs)
        self.ensure_coherence()

    def update_from(self, config_mod, path=''):
        if False:
            i = 10
            return i + 15
        added = config_mod.added
        updated = config_mod.modified
        typechanged = config_mod.typechanged
        self.added &= {join_paths(path, a) for a in added}
        self.modified |= {join_paths(path, u) for u in updated}
        self.typechanged.update({join_paths(path, k): v for (k, v) in typechanged.items()})
        self.ensure_coherence()
        for (k, v) in config_mod.docs.items():
            if not self.docs.get(k, ''):
                self.docs[k] = v

    def update_add(self, config_mod, path=''):
        if False:
            while True:
                i = 10
        added = config_mod.added
        updated = config_mod.modified
        typechanged = config_mod.typechanged
        self.added |= {join_paths(path, a) for a in added}
        self.modified |= {join_paths(path, u) for u in updated}
        self.typechanged.update({join_paths(path, k): v for (k, v) in typechanged.items()})
        self.docs.update({join_paths(path, k): v for (k, v) in config_mod.docs.items() if path == '' or k != 'seed'})
        self.ensure_coherence()

    def ensure_coherence(self):
        if False:
            i = 10
            return i + 15
        self.modified |= {p for a in self.added for p in iter_prefixes(a)}
        self.modified |= {p for u in self.modified for p in iter_prefixes(u)}
        self.modified |= {p for t in self.typechanged for p in iter_prefixes(t)}
        self.added -= set(self.typechanged.keys())
        self.modified -= set(self.typechanged.keys())
        self.modified -= self.added