""" Standard plug-in to make dill module work for compiled stuff.

"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginDillWorkarounds(NuitkaPluginBase):
    """This is to make dill module work with compiled methods."""
    plugin_name = 'dill-compat'
    plugin_desc = "Required for 'dill' package compatibility."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            for i in range(10):
                print('nop')
        return False

    def createPostModuleLoadCode(self, module):
        if False:
            i = 10
            return i + 15
        full_name = module.getFullName()
        if full_name == 'dill':
            return (self.getPluginDataFileContents('dill-postLoad.py'), 'Extending "dill" for compiled types to be pickle-able as well.')

    @staticmethod
    def getPreprocessorSymbols():
        if False:
            while True:
                i = 10
        return {'_NUITKA_PLUGIN_DILL_ENABLED': '1'}

    def getExtraCodeFiles(self):
        if False:
            return 10
        return {'DillPlugin.c': extra_code}
extra_code = '\n#include "nuitka/prelude.h"\n\nvoid registerDillPluginTables(PyThreadState *tstate, char const *module_name, PyMethodDef *reduce_compiled_function, PyMethodDef *create_compiled_function) {\n    PyObject *function_tables = PyObject_GetAttrString((PyObject *)builtin_module, "compiled_function_tables");\n\n    if (function_tables == NULL) {\n        CLEAR_ERROR_OCCURRED(tstate);\n\n        function_tables = MAKE_DICT_EMPTY();\n        PyObject_SetAttrString((PyObject *)builtin_module, "compiled_function_tables", function_tables);\n    }\n\n    PyObject *funcs = MAKE_TUPLE2_0(PyCFunction_New(reduce_compiled_function, NULL), PyCFunction_New(create_compiled_function, NULL));\n\n    PyDict_SetItemString(function_tables, module_name, funcs);\n}\n\n'