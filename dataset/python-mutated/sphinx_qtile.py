import os
from subprocess import CalledProcessError, run
from qtile_docs.collapsible import CollapsibleNode, CollapsibleSection, depart_collapsible_node, visit_collapsible_node
from qtile_docs.commands import QtileCommands
from qtile_docs.graph import QtileGraph
from qtile_docs.hooks import QtileHooks
from qtile_docs.migrations import QtileMigrations
from qtile_docs.module import QtileModule
from qtile_docs.qtile_class import QtileClass

def generate_keybinding_images():
    if False:
        return 10
    this_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(this_dir, '..'))
    run(['make', '-C', base_dir, 'run-ffibuild'])
    run(['make', '-C', this_dir, 'genkeyimg'])

def generate_widget_screenshots():
    if False:
        print('Hello World!')
    this_dir = os.path.dirname(__file__)
    try:
        run(['make', '-C', this_dir, 'genwidgetscreenshots'], check=True)
    except CalledProcessError:
        raise Exception('Widget screenshots failed to build.')

def setup(app):
    if False:
        print('Hello World!')
    generate_keybinding_images()
    if os.getenv('QTILE_BUILD_SCREENSHOTS', False):
        generate_widget_screenshots()
    else:
        print('Skipping screenshot builds...')
    app.add_directive('qtile_class', QtileClass)
    app.add_directive('qtile_hooks', QtileHooks)
    app.add_directive('qtile_module', QtileModule)
    app.add_directive('qtile_commands', QtileCommands)
    app.add_directive('qtile_graph', QtileGraph)
    app.add_directive('qtile_migrations', QtileMigrations)
    app.add_directive('collapsible', CollapsibleSection)
    app.add_node(CollapsibleNode, html=(visit_collapsible_node, depart_collapsible_node))