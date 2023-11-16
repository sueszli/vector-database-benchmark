from cms.management.commands.subcommands.list import plugin_report
from .base import SubcommandsCommand

class DeleteOrphanedPluginsCommand(SubcommandsCommand):
    help_string = "Delete plugins from the CMSPlugins table that should have instances but don't, and ones for which a corresponding plugin model can no longer be found"
    command_name = 'delete-orphaned-plugins'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Obtains a plugin report -\n        cms.management.commands.subcommands.list.plugin_report - and uses it\n        to delete orphaned plugins from the database, i.e. ones that are no\n        longer installed, and ones that have no corresponding saved plugin\n        instances (as will happen if a plugin is inserted into a placeholder,\n        but not saved).\n        '
        self.stdout.write('Obtaining plugin report\n')
        uninstalled_instances = []
        unsaved_instances = []
        for plugin in plugin_report():
            if not plugin['model']:
                for instance in plugin['instances']:
                    uninstalled_instances.append(instance)
            for instance in plugin['unsaved_instances']:
                unsaved_instances.append(instance)
        if options.get('interactive'):
            confirm = input("\nYou have requested to delete any instances of uninstalled plugins and empty plugin instances.\nThere are %d uninstalled plugins and %d empty plugins.\nAre you sure you want to do this?\nType 'yes' to continue, or 'no' to cancel: " % (len(uninstalled_instances), len(unsaved_instances)))
        else:
            confirm = 'yes'
        if confirm == 'yes':
            self.stdout.write('... deleting any instances of uninstalled plugins and empty plugin instances\n')
            for instance in uninstalled_instances:
                instance.delete()
            for instance in unsaved_instances:
                instance.delete()
            self.stdout.write('Deleted instances of: \n    %s uninstalled plugins  \n    %s plugins with unsaved instances\n' % (len(uninstalled_instances), len(unsaved_instances)))
            self.stdout.write('all done\n')