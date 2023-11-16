class FavoriteQueries(object):
    section_name = 'favorite_queries'
    usage = '\nFavorite Queries are a way to save frequently used queries\nwith a short name.\nExamples:\n\n    # Save a new favorite query.\n    > \\fs simple select * from abc where a is not Null;\n\n    # List all favorite queries.\n    > \\f\n    ╒════════╤═══════════════════════════════════════╕\n    │ Name   │ Query                                 │\n    ╞════════╪═══════════════════════════════════════╡\n    │ simple │ SELECT * FROM abc where a is not NULL │\n    ╘════════╧═══════════════════════════════════════╛\n\n    # Run a favorite query.\n    > \\f simple\n    ╒════════╤════════╕\n    │ a      │ b      │\n    ╞════════╪════════╡\n    │ 日本語  │ 日本語  │\n    ╘════════╧════════╛\n\n    # Delete a favorite query.\n    > \\fd simple\n    simple: Deleted\n'
    instance = None

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self.config = config

    @classmethod
    def from_config(cls, config):
        if False:
            while True:
                i = 10
        return FavoriteQueries(config)

    def list(self):
        if False:
            while True:
                i = 10
        return self.config.get(self.section_name, [])

    def get(self, name):
        if False:
            while True:
                i = 10
        return self.config.get(self.section_name, {}).get(name, None)

    def save(self, name, query):
        if False:
            return 10
        self.config.encoding = 'utf-8'
        if self.section_name not in self.config:
            self.config[self.section_name] = {}
        self.config[self.section_name][name] = query
        self.config.write()

    def delete(self, name):
        if False:
            print('Hello World!')
        try:
            del self.config[self.section_name][name]
        except KeyError:
            return '%s: Not Found.' % name
        self.config.write()
        return '%s: Deleted' % name