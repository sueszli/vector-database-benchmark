from sphinx.domains import Index
from sphinx.domains.std import StandardDomain

class PythonAPIIndex(Index):
    objecttype = 'class'
    name = 'apiindex'
    localname = 'Public API Index'
    shortname = 'classes'

    def generate(self, docnames=None):
        if False:
            while True:
                i = 10
        unsorted_objects = [(refname, entry.docname, entry.objtype) for (refname, entry) in self.domain.data['objects'].items() if entry.objtype in ['class', 'function']]
        objects = sorted(unsorted_objects, key=lambda x: x[0].lower())
        entries = []
        for (refname, docname, objtype) in objects:
            if docnames and docname not in docnames:
                continue
            extra_info = objtype
            display_name = refname
            if objtype == 'function':
                display_name += '()'
            entries.append([display_name, 0, docname, refname, extra_info, '', ''])
        return ([('', entries)], False)

def setup(app):
    if False:
        print('Hello World!')
    app.add_index_to_domain('py', PythonAPIIndex)
    StandardDomain.initial_data['labels']['apiindex'] = ('py-apiindex', '', 'Public API Index')
    StandardDomain.initial_data['anonlabels']['apiindex'] = ('py-apiindex', '')
    return {'parallel_read_safe': True, 'parallel_write_safe': True}