""" Graph optimization states.

These are not the graphs you might be thinking of. This is for rending the
progress of optimization into images.
"""
from nuitka import Options
from nuitka.ModuleRegistry import getDoneModules
from nuitka.Tracing import general
graph = None
computation_counters = {}
progressive = False

def _addModuleGraph(module, desc):
    if False:
        return 10
    module_graph = module.asGraph(graph, desc)
    return module_graph

def onModuleOptimizationStep(module):
    if False:
        for i in range(10):
            print('nop')
    if graph is not None:
        computation_counters[module] = computation_counters.get(module, 0) + 1
        if progressive:
            _addModuleGraph(module, computation_counters[module])

def startGraph():
    if False:
        print('Hello World!')
    global graph
    if Options.shallCreateGraph():
        try:
            from pygraphviz import AGraph
            graph = AGraph(name='Optimization', directed=True)
            graph.layout()
        except ImportError:
            general.sysexit('Cannot import pygraphviz module, no graphing capability.')

def endGraph(output_filename):
    if False:
        while True:
            i = 10
    if graph is not None:
        for module in getDoneModules():
            _addModuleGraph(module, 'final')
        graph.draw(output_filename + '.dot', prog='dot')