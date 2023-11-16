import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.utils as utils
import volatility.plugins.mac.pstasks as pstasks
from volatility.renderers import TreeGrid

class mac_contacts(pstasks.mac_tasks):
    """Gets contact names from Contacts.app"""

    def calculate(self):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        procs = pstasks.mac_tasks.calculate(self)
        for proc in procs:
            space = proc.get_process_address_space()
            for map in proc.get_proc_maps():
                if not (map.get_perms() == 'rw-' and (not map.get_path())):
                    continue
                header = space.zread(map.links.start, 32)
                if 'SQLite format' not in header:
                    continue
                data = space.zread(map.links.start, map.links.end - map.links.start)
                for offset in utils.iterfind(data, ':ABPerson'):
                    person = obj.Object('String', offset=map.links.start + offset, vm=space, encoding='utf8', length=256)
                    yield (proc, person)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Contact', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (proc, person) in data:
            person = str(person)[len(':ABPerson'):]
            items = ' '.join(person.split(' ')[:8])
            yield (0, [str(items)])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for (proc, person) in data:
            person = str(person)[len(':ABPerson'):]
            items = ' '.join(person.split(' ')[:8])
            outfd.write('{0}\n'.format(items))