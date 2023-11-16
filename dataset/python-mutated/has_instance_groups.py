from contextlib import suppress
import awxkit.exceptions as exc

class HasInstanceGroups(object):

    def add_instance_group(self, instance_group):
        if False:
            while True:
                i = 10
        with suppress(exc.NoContent):
            self.related['instance_groups'].post(dict(id=instance_group.id))

    def remove_instance_group(self, instance_group):
        if False:
            for i in range(10):
                print('nop')
        with suppress(exc.NoContent):
            self.related['instance_groups'].post(dict(id=instance_group.id, disassociate=instance_group.id))

    def remove_all_instance_groups(self):
        if False:
            while True:
                i = 10
        for ig in self.related.instance_groups.get().results:
            self.remove_instance_group(ig)