import copy
import os
import sys
"\nDump2HHC.py\n\nConverts an AutoDuck dump into an HTML Help Table Of Contents file.\nTODOs:\nAdd support for merging in non-autoduck'd comments into HTML Help files.\n"

class category:

    def __init__(self, category_defn):
        if False:
            while True:
                i = 10
        self.category_defn = category_defn
        self.id = category_defn.id
        self.name = category_defn.label
        self.dump_file = category_defn.id + '.dump'
        self.modules = {}
        self.objects = {}
        self.overviewTopics = {}
        self.extOverviewTopics = {}
        self.constants = {}

    def process(self):
        if False:
            while True:
                i = 10
        d = self.extOverviewTopics
        for oi in self.category_defn.overviewItems.items:
            top = topic()
            top.name = oi.name
            top.context = 'html/' + oi.href
            top.type = 'topic'
            assert not top.name in d and (not top.name in self.overviewTopics), 'Duplicate named topic detected: ' + top.name
            d[top.name] = top

class topic:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.context = None
        self.name = None
        self.type = None
        self.contains = []

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str({'context': self.context, 'name': self.name, 'contains': self.contains})

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if len(self.contains) > 0:
            return repr({'context': self.context, 'name': self.name, 'contains': self.contains})
        else:
            return repr({'context': self.context, 'name': self.name})

def TopicCmp(a, b):
    if False:
        while True:
            i = 10
    if a.name == b.name:
        return 0
    elif a.name > b.name:
        return 1
    else:
        return -1

def TopicKey(a):
    if False:
        while True:
            i = 10
    return a.name

def parseCategories():
    if False:
        i = 10
        return i + 15
    import document_object
    ret = []
    doc = document_object.GetDocument()
    for defn in doc:
        cat = category(defn)
        cat.process()
        ret.append(cat)
    return ret

def parseTopics(cat, input):
    if False:
        i = 10
        return i + 15
    lTags = ['module', 'object', 'topic', 'const']
    line = input.readline()
    if line == '':
        return
    line = line[:-1]
    fields = line.split('\t')
    while len(fields) > 0:
        assert len(fields) == 3, fields
        top = topic()
        top.name = fields[0]
        top.context = fields[1] + '.html'
        line = input.readline()
        if line == '':
            raise ValueError('incomplete topic!')
        line = line[:-1]
        fields = line.split('\t')
        assert len(fields) == 2
        assert len(fields[0]) == 0
        top.type = fields[1]
        if top.type not in lTags:
            line = input.readline()
            line = line[:-1]
            fields = line.split('\t')
            assert len(fields[0]) == 0 and len(fields[1]) == 0
            if line == '':
                raise ValueError('incomplete topic!')
            line = input.readline()
            if line == '':
                return
            line = line[:-1]
            fields = line.split('\t')
            while len(fields) > 0:
                if len(fields[0]) > 0:
                    break
                line = input.readline()
                if line == '':
                    return
                line = line[:-1]
                fields = line.split('\t')
        else:
            if top.type == 'module':
                d = cat.modules
            elif top.type == 'object':
                d = cat.objects
            elif top.type == 'topic':
                d = cat.overviewTopics
            elif top.type == 'const':
                d = cat.constants
            else:
                raise RuntimeError(f"What is '{top.type}'")
            if top.name in d:
                print(f'Duplicate named {top.type} detected: {top.name}')
            line = input.readline()
            line = line[:-1]
            fields = line.split('\t')
            assert len(fields[0]) == 0 and len(fields[1]) == 0, f'{fields}, {top.name}'
            if line == '':
                raise ValueError('incomplete topic!')
            line = input.readline()
            if line == '':
                return
            line = line[:-1]
            fields = line.split('\t')
            while len(fields) > 0:
                if len(fields[0]) > 0:
                    break
                assert len(fields[0]) == 0 and len(fields[1]) > 0, 'Bogus fields: ' + fields
                top2 = topic()
                top2.type = fields[1]
                line = input.readline()
                if line == '':
                    raise ValueError('incomplete topic!')
                line = line[:-1]
                fields = line.split('\t')
                assert len(fields[0]) == 0 and len(fields[1]) == 0, fields
                if top2.type == 'pymeth':
                    top2.name = fields[2]
                    top2.context = f'{_urlescape(top.name)}__{top2.name}_meth.html'
                elif top2.type == 'prop':
                    top2.name = fields[3]
                    top2.context = f'{_urlescape(top.name)}__{top2.name}_prop.html'
                else:
                    line = input.readline()
                    if line == '':
                        return
                    line = line[:-1]
                    fields = line.split('\t')
                    continue
                top.contains.append(top2)
                line = input.readline()
                if line == '':
                    return
                line = line[:-1]
                fields = line.split('\t')
            d[top.name] = top

def _urlescape(name):
    if False:
        while True:
            i = 10
    'Escape the given name for inclusion in a URL.\n\n    Escaping is done in the manner in which AutoDuck(?) seems to be doing\n    it.\n    '
    name = name.replace(' ', '_').replace('(', '.28').replace(')', '.29')
    return name

def _genCategoryHTMLFromDict(dict, output):
    if False:
        for i in range(10):
            print('nop')
    keys = list(dict.keys())
    keys.sort()
    for key in keys:
        topic = dict[key]
        output.write(f'<LI><A HREF="{topic.context}">{topic.name}</A>\n')

def _genOneCategoryHTML(output_dir, cat, title, suffix, *dicts):
    if False:
        for i in range(10):
            print('nop')
    fname = os.path.join(output_dir, cat.id + suffix + '.html')
    output = open(fname, 'w')
    output.write('<HTML><TITLE>' + title + '</TITLE>\n')
    output.write('<BODY>\n')
    output.write('<H1>' + title + '</H1>\n')
    for dict in dicts:
        _genCategoryHTMLFromDict(dict, output)
    output.write('</BODY></HTML>\n')
    output.close()

def _genCategoryTopic(output_dir, cat, title):
    if False:
        print('Hello World!')
    fname = os.path.join(output_dir, cat.id + '.html')
    output = open(fname, 'w')
    output.write('<HTML><TITLE>' + title + '</TITLE>\n')
    output.write('<BODY>\n')
    output.write('<H1>' + title + '</H1>\n')
    for (subtitle, suffix) in (('Overviews', '_overview'), ('Modules', '_modules'), ('Objects', '_objects')):
        output.write(f'<LI><A HREF="{cat.id}{suffix}.html">{subtitle}</A>\n')
    output.write('</BODY></HTML>\n')
    output.close()

def genCategoryHTML(output_dir, cats):
    if False:
        i = 10
        return i + 15
    for cat in cats:
        _genCategoryTopic(output_dir, cat, cat.name)
        _genOneCategoryHTML(output_dir, cat, 'Overviews', '_overview', cat.extOverviewTopics, cat.overviewTopics)
        _genOneCategoryHTML(output_dir, cat, 'Modules', '_modules', cat.modules)
        _genOneCategoryHTML(output_dir, cat, 'Objects', '_objects', cat.objects)
        _genOneCategoryHTML(output_dir, cat, 'Constants', '_constants', cat.constants)

def _genItemsFromDict(dict, cat, output, target, do_children=1):
    if False:
        for i in range(10):
            print('nop')
    CHM = 'mk:@MSITStore:%s.chm::/' % target
    keys = list(dict.keys())
    keys.sort()
    for k in keys:
        context = dict[k].context
        name = dict[k].name
        output.write('\n        <LI> <OBJECT type="text/sitemap">\n             <param name="Name" value="{name}">\n             <param name="ImageNumber" value="1">\n             <param name="Local" value="{CHM}{context}">\n             </OBJECT>\n      '.format(**locals()))
        if not do_children:
            continue
        if len(dict[k].contains) > 0:
            output.write('<UL>')
        containees = copy.copy(dict[k].contains)
        containees.sort(key=TopicKey)
        for m in containees:
            output.write(f'\n        <LI><OBJECT type="text/sitemap">\n             <param name="Name" value="{m.name}">\n             <param name="ImageNumber" value="11">\n             <param name="Local" value="{CHM}{m.context}">\n            </OBJECT>')
        if len(dict[k].contains) > 0:
            output.write('\n        </UL>')

def genTOC(cats, output, title, target):
    if False:
        for i in range(10):
            print('nop')
    CHM = 'mk:@MSITStore:%s.chm::/' % target
    output.write('\n<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">\n<HTML>\n<HEAD>\n<meta name="GENERATOR" content="Microsoft&reg; HTML Help Workshop 4.1">\n<!-- Sitemap 1.0 -->\n</HEAD><BODY>\n<OBJECT type="text/site properties">\n    <param name="ImageType" value="Folder">\n</OBJECT>\n<UL>\n    <LI> <OBJECT type="text/sitemap">\n        <param name="Name" value="{title}">\n        <param name="ImageNumber" value="1">\n        <param name="Local" value="{CHM}{target}.html">\n        </OBJECT>\n    <UL>\n'.format(**locals()))
    for cat in cats:
        cat_name = cat.name
        cat_id = cat.id
        output.write('            <LI> <OBJECT type="text/sitemap">\n                 <param name="Name" value="{cat_name}">\n                 <param name="ImageNumber" value="1">\n                 <param name="Local" value="{CHM}{cat_id}.html">\n                 </OBJECT>\n            <UL>\n        '.format(**locals()))
        output.write('                <LI> <OBJECT type="text/sitemap">\n                     <param name="Name" value="Overviews">\n                     <param name="ImageNumber" value="1">\n                     <param name="Local" value="{CHM}{cat_id}_overview.html">\n                     </OBJECT>\n                <UL>\n        '.format(**locals()))
        _genItemsFromDict(cat.overviewTopics, cat, output, target)
        _genItemsFromDict(cat.extOverviewTopics, cat, output, target)
        output.write('\n                </UL>')
        output.write('\n                <LI> <OBJECT type="text/sitemap">\n                    <param name="Name" value="Modules">\n                    <param name="ImageNumber" value="1">\n                    <param name="Local" value="{CHM}{cat_id}_modules.html">\n                    </OBJECT>\n                <UL>\n'.format(**locals()))
        _genItemsFromDict(cat.modules, cat, output, target)
        output.write('\n                </UL>')
        output.write('\n                <LI> <OBJECT type="text/sitemap">\n                    <param name="Name" value="Objects">\n                    <param name="ImageNumber" value="1">\n                    <param name="Local" value="{CHM}{cat_id}_objects.html">\n                    </OBJECT>\n                <UL>'.format(**locals()))
        _genItemsFromDict(cat.objects, cat, output, target, do_children=0)
        output.write('\n                </UL>')
        output.write('\n    <LI> <OBJECT type="text/sitemap">\n         <param name="Name" value="Constants">\n         <param name="ImageNumber" value="1">\n         <param name="Local" value="{CHM}{cat_id}_constants.html">\n         </OBJECT>\n           <UL>\n'.format(**locals()))
        _genItemsFromDict(cat.constants, cat, output, target)
        output.write('\n           </UL>')
        output.write('\n        </UL>')
    output.write('\n</UL>\n</BODY></HTML>\n')

def main():
    if False:
        i = 10
        return i + 15
    gen_dir = sys.argv[1]
    cats = parseCategories()
    for cat in cats:
        file = os.path.join(gen_dir, cat.dump_file)
        input = open(file, 'r')
        parseTopics(cat, input)
        input.close()
    output = open(sys.argv[2], 'w')
    genTOC(cats, output, sys.argv[3], sys.argv[4])
    genCategoryHTML(gen_dir, cats)
if __name__ == '__main__':
    main()