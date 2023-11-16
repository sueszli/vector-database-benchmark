""" Multidist re-formulation. """
import os
from nuitka.Options import getMainEntryPointFilenames
from nuitka.utils.ModuleNames import makeMultidistModuleName

def _stripPythonSuffix(filename):
    if False:
        return 10
    if filename.lower().endswith('.py'):
        return filename[:-3]
    elif filename.lower().endswith('.pyw'):
        return filename[:-4]
    else:
        return filename

def createMultidistMainSourceCode(main_filenames):
    if False:
        print('Hello World!')
    main_basenames = [_stripPythonSuffix(os.path.basename(main_filename)) for main_filename in main_filenames]
    main_module_names = [makeMultidistModuleName(count, main_basename) for (count, main_basename) in enumerate(main_basenames, start=1)]
    from nuitka.utils.Jinja2 import renderTemplateFromString
    source_code = renderTemplateFromString('\nimport sys, re, os\nmain_basename = re.sub(r\'(.pyw?|\\.exe|\\.bin)?$\', \'\', os.path.normcase(os.path.basename(sys.argv[0])))\n{% for main_module_name, main_basename in zip(main_module_names, main_basenames) %}\nif main_basename == "{{main_basename}}":\n    __import__("{{main_module_name.asString()}}")\n    sys.exit(0)\n{% endfor %}\n\nsys.exit("Error, failed to detect what to do for filename derived name \'%s\'." % main_basename)\n', main_module_names=main_module_names, main_basenames=main_basenames, zip=zip)
    return source_code

def locateMultidistModule(module_name):
    if False:
        print('Hello World!')
    multidist_index = int(str(module_name).split('-')[1])
    return (module_name, getMainEntryPointFilenames()[multidist_index - 1], 'py', 'absolute')