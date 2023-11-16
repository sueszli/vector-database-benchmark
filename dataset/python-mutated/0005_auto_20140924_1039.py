from django.db import migrations
from django.db.models import F
from treebeard.numconv import NumConv
STEPLEN = 4
ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class MP_AddHandler:

    def __init__(self):
        if False:
            print('Hello World!')
        self.stmts = []
NUM = NumConv(len(ALPHABET), ALPHABET)

def _int2str(num):
    if False:
        return 10
    return NUM.int2str(num)

def _str2int(num):
    if False:
        i = 10
        return i + 15
    return NUM.str2int(num)

def _get_basepath(path, depth):
    if False:
        print('Hello World!')
    ':returns: The base path of another path up to a given depth'
    if path:
        return path[0:depth * STEPLEN]
    return ''

def _get_path(path, depth, newstep):
    if False:
        i = 10
        return i + 15
    '\n    Builds a path given some values\n\n    :param path: the base path\n    :param depth: the depth of the  node\n    :param newstep: the value (integer) of the new step\n    '
    parentpath = _get_basepath(path, depth - 1)
    key = _int2str(newstep)
    return '{0}{1}{2}'.format(parentpath, ALPHABET[0] * (STEPLEN - len(key)), key)

def _inc_path(obj):
    if False:
        print('Hello World!')
    ':returns: The path of the next sibling of a given node path.'
    newpos = _str2int(obj.path[-STEPLEN:]) + 1
    key = _int2str(newpos)
    if len(key) > STEPLEN:
        raise Exception("Path Overflow from: '%s'" % (obj.path,))
    return '{0}{1}{2}'.format(obj.path[:-STEPLEN], ALPHABET[0] * (STEPLEN - len(key)), key)

class MP_AddRootHandler(MP_AddHandler):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.kwargs = kwargs

    def process(self):
        if False:
            for i in range(10):
                print('nop')
        last_root = self.kwargs['last_root']
        if last_root:
            newpath = _inc_path(last_root)
        else:
            newpath = _get_path(None, 1, 1)
        newobj = self.kwargs['instance']
        newobj.depth = 1
        newobj.path = newpath
        newobj.save()
        return newobj

class MP_AddChildHandler(MP_AddHandler):

    def __init__(self, node, model, **kwargs):
        if False:
            return 10
        super().__init__()
        self.node = node
        self.node_cls = node.__class__
        self.kwargs = kwargs
        self.model = model

    def process(self):
        if False:
            print('Hello World!')
        newobj = self.kwargs['instance']
        newobj.depth = self.node.depth + 1
        if self.node.numchild == 0:
            newobj.path = _get_path(self.node.path, newobj.depth, 1)
            max_length = self.node_cls._meta.get_field('path').max_length
            if len(newobj.path) > max_length:
                raise Exception('The new node is too deep in the tree, try increasing the path.max_length property and UPDATE your database')
        else:
            newobj.path = _inc_path(self.node.last_child)
        newobj.save()
        newobj._cached_parent_obj = self.node
        self.model.objects.filter(path=self.node.path).update(numchild=F('numchild') + 1)
        self.node.numchild += 1
        return newobj

def move_to_mp(apps, schema_editor):
    if False:
        print('Hello World!')
    db_alias = schema_editor.connection.alias
    Page = apps.get_model('cms', 'Page')
    CMSPlugin = apps.get_model('cms', 'CMSPlugin')
    pages = Page.objects.using(db_alias).order_by('tree_id', 'level', 'lft')
    cache = {}
    last_root = None
    for page in pages:
        if not page.parent_id:
            handler = MP_AddRootHandler(instance=page, last_root=last_root)
            handler.process()
            last_root = page
            page.last_child = None
        else:
            parent = cache[page.parent_id]
            handler = MP_AddChildHandler(parent, Page, instance=page)
            handler.process()
            parent.last_child = page
        cache[page.pk] = page
    plugins = CMSPlugin.objects.using(db_alias).order_by('tree_id', 'level', 'lft')
    cache = {}
    last_root = None
    for plugin in plugins:
        if not plugin.parent_id:
            handler = MP_AddRootHandler(instance=plugin, last_root=last_root)
            handler.process()
            last_root = plugin
            plugin.last_child = None
        else:
            parent = cache[plugin.parent_id]
            handler = MP_AddChildHandler(parent, CMSPlugin, instance=plugin)
            handler.process()
            parent.last_child = plugin
        cache[plugin.pk] = plugin

class Migration(migrations.Migration):
    dependencies = [('cms', '0004_auto_20140924_1038')]
    operations = [migrations.RunPython(move_to_mp)]