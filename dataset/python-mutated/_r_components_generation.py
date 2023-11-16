import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
r_component_string = "#' @export\n{funcname} <- function({default_argtext}{wildcards}) {{\n    {wildcard_declaration}\n    props <- list({default_paramtext}{wildcards})\n    if (length(props) > 0) {{\n        props <- props[!vapply(props, is.null, logical(1))]\n    }}\n    component <- list(\n        props = props,\n        type = '{name}',\n        namespace = '{project_shortname}',\n        propNames = c({prop_names}{wildcard_names}),\n        package = '{package_name}'\n        )\n\n    structure(component, class = c('dash_component', 'list'))\n}}\n"
frame_open_template = '.{rpkgname}_js_metadata <- function() {{\ndeps_metadata <- list('
frame_element_template = '`{dep_name}` = structure(list(name = "{dep_name}",\nversion = "{project_ver}", src = list(href = NULL,\nfile = "deps"), meta = NULL,\nscript = {script_name},\nstylesheet = {css_name}, head = NULL, attachment = NULL, package = "{rpkgname}",\nall_files = FALSE{async_or_dynamic}), class = "html_dependency")'
frame_body_template = '`{project_shortname}` = structure(list(name = "{project_shortname}",\nversion = "{project_ver}", src = list(href = NULL,\nfile = "deps"), meta = NULL,\nscript = {script_name},\nstylesheet = {css_name}, head = NULL, attachment = NULL, package = "{rpkgname}",\nall_files = FALSE{async_or_dynamic}), class = "html_dependency")'
frame_close_template = ')\nreturn(deps_metadata)\n}\n'
help_string = '% Auto-generated: do not edit by hand\n\\name{{{funcname}}}\n\n\\alias{{{funcname}}}\n\n\\title{{{name} component}}\n\n\\description{{\n{description}\n}}\n\n\\usage{{\n{funcname}({default_argtext})\n}}\n\n\\arguments{{\n{item_text}\n}}\n\n\\value{{{value_text}}}\n\n'
description_template = 'Package: {package_name}\nTitle: {package_title}\nVersion: {package_version}\nDescription: {package_description}\nDepends: R (>= 3.0.2){package_depends}\nImports: {package_imports}\nSuggests: {package_suggests}{package_rauthors}\nLicense: {package_license}{package_copyright}\nURL: {package_url}\nBugReports: {package_issues}\nEncoding: UTF-8\nLazyData: true{vignette_builder}\nKeepSource: true\n'
rbuild_ignore_string = '# ignore JS config files/folders\nnode_modules/\ncoverage/\nsrc/\nlib/\n.babelrc\n.builderrc\n.eslintrc\n.npmignore\n.editorconfig\n.eslintignore\n.prettierrc\n.circleci\n.github\n\n# demo folder has special meaning in R\n# this should hopefully make it still\n# allow for the possibility to make R demos\ndemo/.*\\.js\ndemo/.*\\.html\ndemo/.*\\.css\n\n# ignore Python files/folders\nsetup.py\nusage.py\nsetup.py\nrequirements.txt\nMANIFEST.in\nCHANGELOG.md\ntest/\n# CRAN has weird LICENSE requirements\nLICENSE.txt\n^.*\\.Rproj$\n^\\.Rproj\\.user$\n'
pkghelp_stub = '% Auto-generated: do not edit by hand\n\\docType{{package}}\n\\name{{{package_name}-package}}\n\\alias{{{package_name}}}\n\\title{{{pkg_help_title}}}\n\\description{{\n{pkg_help_description}\n}}\n\\author{{\n\\strong{{Maintainer}}: {maintainer}\n}}\n'
wildcard_helper = '\ndash_assert_valid_wildcards <- function (attrib = list("data", "aria"), ...)\n{\n    args <- list(...)\n    validation_results <- lapply(names(args), function(x) {\n        grepl(paste0("^(", paste0(attrib, collapse="|"), ")-[a-zA-Z0-9_-]+$"),\n            x)\n    })\n    if (FALSE %in% validation_results) {\n        stop(sprintf("The following props are not valid in this component: \'%s\'",\n            paste(names(args)[grepl(FALSE, unlist(validation_results))],\n                collapse = ", ")), call. = FALSE)\n    }\n    else {\n        return(args)\n    }\n}\n'
wildcard_template = '\n    wildcard_names = names(dash_assert_valid_wildcards(attrib = list({}), ...))\n'
wildcard_help_template = '\n\n\n\\item{{...}}{{wildcards allowed have the form: `{}`}}\n'

def generate_class_string(name, props, project_shortname, prefix):
    if False:
        print('Hello World!')
    package_name = snake_case_to_camel_case(project_shortname)
    props = reorder_props(props=props)
    prop_keys = list(props.keys())
    wildcards = ''
    wildcard_declaration = ''
    wildcard_names = ''
    default_paramtext = ''
    default_argtext = ''
    accepted_wildcards = ''
    if any((key.endswith('-*') for key in prop_keys)):
        accepted_wildcards = get_wildcards_r(prop_keys)
        wildcards = ', ...'
        wildcard_declaration = wildcard_template.format(accepted_wildcards.replace('-*', ''))
        wildcard_names = ', wildcard_names'
    prop_names = ', '.join(("'{}'".format(p) for p in prop_keys if '*' not in p and p not in ['setProps']))
    for item in prop_keys[:]:
        if item.endswith('-*') or item == 'setProps':
            prop_keys.remove(item)
        elif item in r_keywords:
            prop_keys.remove(item)
            warnings.warn('WARNING: prop "{}" in component "{}" is an R keyword - REMOVED FROM THE R COMPONENT'.format(item, name))
    default_argtext += ', '.join(('{}=NULL'.format(p) for p in prop_keys))
    default_paramtext += ', '.join(('{0}={0}'.format(p) if p != 'children' else '{}=children'.format(p) for p in prop_keys))
    return r_component_string.format(funcname=format_fn_name(prefix, name), name=name, default_argtext=default_argtext, wildcards=wildcards, wildcard_declaration=wildcard_declaration, default_paramtext=default_paramtext, project_shortname=project_shortname, prop_names=prop_names, wildcard_names=wildcard_names, package_name=package_name)

def generate_js_metadata(pkg_data, project_shortname):
    if False:
        while True:
            i = 10
    'Dynamically generate R function to supply JavaScript and CSS dependency\n    information required by the dash package for R.\n\n    Parameters\n    ----------\n    project_shortname = component library name, in snake case\n\n    Returns\n    -------\n    function_string = complete R function code to provide component features\n    '
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(project_shortname)
    alldist = getattr(mod, '_js_dist', []) + getattr(mod, '_css_dist', [])
    project_ver = pkg_data.get('version')
    rpkgname = snake_case_to_camel_case(project_shortname)
    function_frame_open = frame_open_template.format(rpkgname=rpkgname)
    function_frame = []
    function_frame_body = []
    if len(alldist) > 1:
        for dep in range(len(alldist)):
            curr_dep = alldist[dep]
            rpp = curr_dep['relative_package_path']
            async_or_dynamic = get_async_type(curr_dep)
            if 'dash_' in rpp:
                dep_name = rpp.split('.')[0]
            else:
                dep_name = '{}'.format(project_shortname)
            if 'css' in rpp:
                css_name = "'{}'".format(rpp)
                script_name = 'NULL'
            else:
                script_name = "'{}'".format(rpp)
                css_name = 'NULL'
            function_frame += [frame_element_template.format(dep_name=dep_name, project_ver=project_ver, rpkgname=rpkgname, project_shortname=project_shortname, script_name=script_name, css_name=css_name, async_or_dynamic=async_or_dynamic)]
            function_frame_body = ',\n'.join(function_frame)
    elif len(alldist) == 1:
        dep = alldist[0]
        rpp = dep['relative_package_path']
        async_or_dynamic = get_async_type(dep)
        if 'css' in rpp:
            css_name = "'{}'".format(rpp)
            script_name = 'NULL'
        else:
            script_name = "'{}'".format(rpp)
            css_name = 'NULL'
        function_frame_body = frame_body_template.format(project_shortname=project_shortname, project_ver=project_ver, rpkgname=rpkgname, script_name=script_name, css_name=css_name, async_or_dynamic=async_or_dynamic)
    function_string = ''.join([function_frame_open, function_frame_body, frame_close_template])
    return function_string

def get_async_type(dep):
    if False:
        print('Hello World!')
    async_or_dynamic = ''
    for key in dep.keys():
        if key in ['async', 'dynamic']:
            keyval = dep[key]
            if not isinstance(keyval, bool):
                keyval = "'{}'".format(keyval.lower())
            else:
                keyval = str(keyval).upper()
            async_or_dynamic = ', {} = {}'.format(key, keyval)
    return async_or_dynamic

def wrap(tag, code):
    if False:
        return 10
    if tag == '':
        return code
    return '\\{}{{\n{}}}'.format(tag, code)

def write_help_file(name, props, description, prefix, rpkg_data):
    if False:
        print('Hello World!')
    "Write R documentation file (.Rd) given component name and properties.\n\n    Parameters\n    ----------\n    name = the name of the Dash component for which a help file is generated\n    props = the properties of the component\n    description = the component's description, inserted into help file header\n    prefix = the DashR library prefix (optional, can be a blank string)\n    rpkg_data = package metadata (optional)\n\n    Returns\n    -------\n    writes an R help file to the man directory for the generated R package\n    "
    funcname = format_fn_name(prefix, name)
    file_name = funcname + '.Rd'
    wildcards = ''
    default_argtext = ''
    item_text = ''
    accepted_wildcards = ''
    value_text = 'named list of JSON elements corresponding to React.js properties and their values'
    prop_keys = list(props.keys())
    if any((key.endswith('-*') for key in prop_keys)):
        accepted_wildcards = get_wildcards_r(prop_keys)
        wildcards = ', ...'
    for item in prop_keys[:]:
        if item.endswith('-*') or item in r_keywords or item == 'setProps':
            prop_keys.remove(item)
    default_argtext += ', '.join(('{}=NULL'.format(p) for p in prop_keys))
    item_text += '\n\n'.join(('\\item{{{}}}{{{}{}}}'.format(p, print_r_type(props[p]['type']), props[p]['description']) for p in prop_keys))
    description = re.sub('(?<!\\\\)%', '\\%', description)
    item_text = re.sub('(?<!\\\\)%', '\\%', item_text)
    if '**Example Usage**' in description:
        description = description.split('**Example Usage**')[0].rstrip()
    if wildcards == ', ...':
        default_argtext += wildcards
        item_text += wildcard_help_template.format(accepted_wildcards)
    file_path = os.path.join('man', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(help_string.format(funcname=funcname, name=name, default_argtext=textwrap.fill(default_argtext, width=60, break_long_words=False), item_text=item_text, value_text=value_text, description=description.replace('\n', ' ')))
    if rpkg_data is not None and 'r_examples' in rpkg_data:
        ex = rpkg_data.get('r_examples')
        the_ex = ([e for e in ex if e.get('name') == funcname] or [None])[0]
        result = ''
        if the_ex and 'code' in the_ex.keys():
            result += wrap('examples', wrap('dontrun' if the_ex.get('dontrun') else '', the_ex['code']))
            with open(file_path, 'a+', encoding='utf-8') as fa:
                fa.write(result + '\n')

def write_class_file(name, props, description, project_shortname, prefix=None, rpkg_data=None):
    if False:
        return 10
    props = reorder_props(props=props)
    write_help_file(name, props, description, prefix, rpkg_data)
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n'
    class_string = generate_class_string(name, props, project_shortname, prefix)
    file_name = format_fn_name(prefix, name) + '.R'
    file_path = os.path.join('R', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(import_string)
        f.write(class_string)
    print('Generated {}'.format(file_name))

def write_js_metadata(pkg_data, project_shortname, has_wildcards):
    if False:
        return 10
    'Write an internal (not exported) R function to return all JS\n    dependencies as required by dash.\n\n    Parameters\n    ----------\n    project_shortname = hyphenated string, e.g. dash-html-components\n\n    Returns\n    -------\n    '
    function_string = generate_js_metadata(pkg_data=pkg_data, project_shortname=project_shortname)
    file_name = 'internal.R'
    if not os.path.exists('R'):
        os.makedirs('R')
    file_path = os.path.join('R', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(function_string)
        if has_wildcards:
            f.write(wildcard_helper)
    if os.path.exists('inst/deps'):
        shutil.rmtree('inst/deps')
    os.makedirs('inst/deps')
    for (rel_dirname, _, filenames) in os.walk(project_shortname):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension in ['.py', '.pyc', '.json']:
                continue
            target_dirname = os.path.join(os.path.join('inst/deps/', os.path.relpath(rel_dirname, project_shortname)))
            if not os.path.exists(target_dirname):
                os.makedirs(target_dirname)
            shutil.copy(os.path.join(rel_dirname, filename), target_dirname)

def generate_rpkg(pkg_data, rpkg_data, project_shortname, export_string, package_depends, package_imports, package_suggests, has_wildcards):
    if False:
        print('Hello World!')
    'Generate documents for R package creation.\n\n    Parameters\n    ----------\n    pkg_data\n    rpkg_data\n    project_shortname\n    export_string\n    package_depends\n    package_imports\n    package_suggests\n    has_wildcards\n\n    Returns\n    -------\n    '
    package_name = snake_case_to_camel_case(project_shortname)
    package_copyright = ''
    package_rauthors = ''
    lib_name = pkg_data.get('name')
    if rpkg_data is not None:
        if rpkg_data.get('pkg_help_title'):
            package_title = rpkg_data.get('pkg_help_title', pkg_data.get('description', ''))
        if rpkg_data.get('pkg_help_description'):
            package_description = rpkg_data.get('pkg_help_description', pkg_data.get('description', ''))
        if rpkg_data.get('pkg_copyright'):
            package_copyright = '\nCopyright: {}'.format(rpkg_data.get('pkg_copyright', ''))
    else:
        package_title = pkg_data.get('description', '')
        package_description = pkg_data.get('description', '')
    package_version = pkg_data.get('version', '0.0.1')
    if package_depends:
        package_depends = ', ' + package_depends.strip(',').lstrip()
        package_depends = re.sub('(,(?![ ]))', ', ', package_depends)
    if package_imports:
        package_imports = package_imports.strip(',').lstrip()
        package_imports = re.sub('(,(?![ ]))', ', ', package_imports)
    if package_suggests:
        package_suggests = package_suggests.strip(',').lstrip()
        package_suggests = re.sub('(,(?![ ]))', ', ', package_suggests)
    if 'bugs' in pkg_data:
        package_issues = pkg_data['bugs'].get('url', '')
    else:
        package_issues = ''
        print('Warning: a URL for bug reports was not provided. Empty string inserted.', file=sys.stderr)
    if 'homepage' in pkg_data:
        package_url = pkg_data.get('homepage', '')
    else:
        package_url = ''
        print('Warning: a homepage URL was not provided. Empty string inserted.', file=sys.stderr)
    package_author = pkg_data.get('author')
    package_author_name = package_author.split(' <')[0]
    package_author_email = package_author.split(' <')[1][:-1]
    package_author_fn = package_author_name.split(' ')[0]
    package_author_ln = package_author_name.rsplit(' ', 2)[-1]
    maintainer = pkg_data.get('maintainer', pkg_data.get('author'))
    if '<' not in package_author:
        print('Error, aborting R package generation: R packages require a properly formatted author field or installation will fail. Please include an email address enclosed within < > brackets in package.json. ', file=sys.stderr)
        sys.exit(1)
    if rpkg_data is not None:
        if rpkg_data.get('pkg_authors'):
            package_rauthors = '\nAuthors@R: {}'.format(rpkg_data.get('pkg_authors', ''))
        else:
            package_rauthors = '\nAuthors@R: person("{}", "{}", role = c("aut", "cre"), email = "{}")'.format(package_author_fn, package_author_ln, package_author_email)
    if not (os.path.isfile('LICENSE') or os.path.isfile('LICENSE.txt')):
        package_license = pkg_data.get('license', '')
    else:
        package_license = pkg_data.get('license', '') + ' + file LICENSE'
        if not os.path.isfile('LICENSE'):
            os.symlink('LICENSE.txt', 'LICENSE')
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n'
    packages_string = ''
    rpackage_list = package_depends.split(', ') + package_imports.split(', ')
    rpackage_list = filter(bool, rpackage_list)
    if rpackage_list:
        for rpackage in rpackage_list:
            packages_string += '\nimport({})\n'.format(rpackage)
    if os.path.exists('vignettes'):
        vignette_builder = '\nVignetteBuilder: knitr'
        if 'knitr' not in package_suggests and 'rmarkdown' not in package_suggests:
            package_suggests += ', knitr, rmarkdown'
            package_suggests = package_suggests.lstrip(', ')
    else:
        vignette_builder = ''
    pkghelp_stub_path = os.path.join('man', package_name + '-package.Rd')
    write_js_metadata(pkg_data, project_shortname, has_wildcards)
    with open('NAMESPACE', 'w+', encoding='utf-8') as f:
        f.write(import_string)
        f.write(export_string)
        f.write(packages_string)
    with open('.Rbuildignore', 'w+', encoding='utf-8') as f2:
        f2.write(rbuild_ignore_string)
    description_string = description_template.format(package_name=package_name, package_title=package_title, package_description=package_description, package_version=package_version, package_rauthors=package_rauthors, package_depends=package_depends, package_imports=package_imports, package_suggests=package_suggests, package_license=package_license, package_copyright=package_copyright, package_url=package_url, package_issues=package_issues, vignette_builder=vignette_builder)
    with open('DESCRIPTION', 'w+', encoding='utf-8') as f3:
        f3.write(description_string)
    if rpkg_data is not None:
        if rpkg_data.get('pkg_help_description'):
            pkghelp = pkghelp_stub.format(package_name=package_name, pkg_help_title=rpkg_data.get('pkg_help_title'), pkg_help_description=rpkg_data.get('pkg_help_description'), lib_name=lib_name, maintainer=maintainer)
            with open(pkghelp_stub_path, 'w', encoding='utf-8') as f4:
                f4.write(pkghelp)

def snake_case_to_camel_case(namestring):
    if False:
        while True:
            i = 10
    s = namestring.split('_')
    return s[0] + ''.join((w.capitalize() for w in s[1:]))

def format_fn_name(prefix, name):
    if False:
        print('Hello World!')
    if prefix:
        return prefix + snake_case_to_camel_case(name)
    return snake_case_to_camel_case(name[0].lower() + name[1:])

def generate_exports(project_shortname, components, metadata, pkg_data, rpkg_data, prefix, package_depends, package_imports, package_suggests, **kwargs):
    if False:
        print('Hello World!')
    export_string = make_namespace_exports(components, prefix)
    has_wildcards = False
    for component_data in metadata.values():
        if any((key.endswith('-*') for key in component_data['props'])):
            has_wildcards = True
            break
    generate_rpkg(pkg_data, rpkg_data, project_shortname, export_string, package_depends, package_imports, package_suggests, has_wildcards)

def make_namespace_exports(components, prefix):
    if False:
        i = 10
        return i + 15
    export_string = ''
    for component in components:
        if not component.endswith('-*') and str(component) not in r_keywords and (str(component) not in ['setProps', 'children']):
            export_string += 'export({}{})\n'.format(prefix, component)
    rfilelist = []
    omitlist = ['utils.R', 'internal.R'] + ['{}{}.R'.format(prefix, component) for component in components]
    fnlist = []
    for script in os.listdir('R'):
        if script.endswith('.R') and script not in omitlist:
            rfilelist += [os.path.join('R', script)]
    for rfile in rfilelist:
        with open(rfile, 'r', encoding='utf-8') as script:
            s = script.read()
            s = re.sub('#.*$', '', s, flags=re.M)
            s = s.replace('\n', ' ').replace('\r', ' ')
            s = re.sub("'([^'\\\\]|\\\\'|\\\\[^'])*'", "''", s)
            s = re.sub('"([^"\\\\]|\\\\"|\\\\[^"])*"', '""', s)
            prev_len = len(s) + 1
            while len(s) < prev_len:
                prev_len = len(s)
                s = re.sub('\\(([^()]|\\(\\))*\\)', '()', s)
                s = re.sub('\\{([^{}]|\\{\\})*\\}', '{}', s)
            matches = re.findall('([^A-Za-z0-9._]|^)([A-Za-z0-9._]+)\\s*(=|<-)\\s*function', s)
            for match in matches:
                fn = match[1]
                if fn[0] != '.' and fn not in fnlist:
                    fnlist.append(fn)
    export_string += '\n'.join(('export({})'.format(function) for function in fnlist))
    return export_string

def get_r_prop_types(type_object):
    if False:
        for i in range(10):
            print('nop')
    'Mapping from the PropTypes js type object to the R type.'

    def shape_or_exact():
        if False:
            return 10
        return 'lists containing elements {}.\n{}'.format(', '.join(("'{}'".format(t) for t in type_object['value'])), 'Those elements have the following types:\n{}'.format('\n'.join((create_prop_docstring_r(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), indent_num=1) for (prop_name, prop) in type_object['value'].items()))))
    return dict(array=lambda : 'unnamed list', bool=lambda : 'logical', number=lambda : 'numeric', string=lambda : 'character', object=lambda : 'named list', any=lambda : 'logical | numeric | character | named list | unnamed list', element=lambda : 'dash component', node=lambda : 'a list of or a singular dash component, string or number', enum=lambda : 'a value equal to: {}'.format(', '.join(('{}'.format(str(t['value'])) for t in type_object['value']))), union=lambda : '{}'.format(' | '.join(('{}'.format(get_r_type(subType)) for subType in type_object['value'] if get_r_type(subType) != ''))), arrayOf=lambda : 'list' + (' of {}s'.format(get_r_type(type_object['value'])) if get_r_type(type_object['value']) != '' else ''), objectOf=lambda : 'list with named elements and values of type {}'.format(get_r_type(type_object['value'])), shape=shape_or_exact, exact=shape_or_exact)

def get_r_type(type_object, is_flow_type=False, indent_num=0):
    if False:
        print('Hello World!')
    '\n    Convert JS types to R types for the component definition\n    Parameters\n    ----------\n    type_object: dict\n        react-docgen-generated prop type dictionary\n    is_flow_type: bool\n    indent_num: int\n        Number of indents to use for the docstring for the prop\n    Returns\n    -------\n    str\n        Python type string\n    '
    js_type_name = type_object['name']
    js_to_r_types = get_r_prop_types(type_object=type_object)
    if 'computed' in type_object and type_object['computed'] or type_object.get('type', '') == 'function':
        return ''
    if js_type_name in js_to_r_types:
        prop_type = js_to_r_types[js_type_name]()
        return prop_type
    return ''

def print_r_type(typedata):
    if False:
        for i in range(10):
            print('nop')
    typestring = get_r_type(typedata).capitalize()
    if typestring:
        typestring += '. '
    return typestring

def create_prop_docstring_r(prop_name, type_object, required, description, indent_num, is_flow_type=False):
    if False:
        i = 10
        return i + 15
    '\n    Create the Dash component prop docstring\n    Parameters\n    ----------\n    prop_name: str\n        Name of the Dash component prop\n    type_object: dict\n        react-docgen-generated prop type dictionary\n    required: bool\n        Component is required?\n    description: str\n        Dash component description\n    indent_num: int\n        Number of indents to use for the context block\n        (creates 2 spaces for every indent)\n    is_flow_type: bool\n        Does the prop use Flow types? Otherwise, uses PropTypes\n    Returns\n    -------\n    str\n        Dash component prop docstring\n    '
    r_type_name = get_r_type(type_object=type_object, is_flow_type=is_flow_type, indent_num=indent_num + 1)
    indent_spacing = '  ' * indent_num
    if '\n' in r_type_name:
        return '{indent_spacing}- {name} ({is_required}): {description}. {name} has the following type: {type}'.format(indent_spacing=indent_spacing, name=prop_name, type=r_type_name, description=description, is_required='required' if required else 'optional')
    return '{indent_spacing}- {name} ({type}{is_required}){description}'.format(indent_spacing=indent_spacing, name=prop_name, type='{}; '.format(r_type_name) if r_type_name else '', description=': {}'.format(description) if description != '' else '', is_required='required' if required else 'optional')

def get_wildcards_r(prop_keys):
    if False:
        return 10
    wildcards = ''
    wildcards += ', '.join(("'{}'".format(p) for p in prop_keys if p.endswith('-*')))
    if wildcards == '':
        wildcards = 'NULL'
    return wildcards