"""SCons.Tool.packaging.msi

The msi packager.
"""
__revision__ = '__FILE__ __REVISION__ __DATE__ __DEVELOPER__'
import os
import SCons
from SCons.Action import Action
from SCons.Builder import Builder
from xml.dom.minidom import Document
from xml.sax.saxutils import escape
from SCons.Tool.packaging import stripinstallbuilder

def convert_to_id(s, id_set):
    if False:
        for i in range(10):
            print('nop')
    ' Some parts of .wxs need an Id attribute (for example: The File and\n    Directory directives. The charset is limited to A-Z, a-z, digits,\n    underscores, periods. Each Id must begin with a letter or with a\n    underscore. Google for "CNDL0015" for information about this.\n\n    Requirements:\n     * the string created must only contain chars from the target charset.\n     * the string created must have a minimal editing distance from the\n       original string.\n     * the string created must be unique for the whole .wxs file.\n\n    Observation:\n     * There are 62 chars in the charset.\n\n    Idea:\n     * filter out forbidden characters. Check for a collision with the help\n       of the id_set. Add the number of the number of the collision at the\n       end of the created string. Furthermore care for a correct start of\n       the string.\n    '
    charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz0123456789_.'
    if s[0] in '0123456789.':
        s = '_' + s
    id = ''.join([c for c in s if c in charset])
    try:
        return id_set[id][s]
    except KeyError:
        if id not in id_set:
            id_set[id] = {s: id}
        else:
            id_set[id][s] = id + str(len(id_set[id]))
        return id_set[id][s]

def is_dos_short_file_name(file):
    if False:
        for i in range(10):
            print('nop')
    ' Examine if the given file is in the 8.3 form.\n    '
    (fname, ext) = os.path.splitext(file)
    proper_ext = len(ext) == 0 or 2 <= len(ext) <= 4
    proper_fname = file.isupper() and len(fname) <= 8
    return proper_ext and proper_fname

def gen_dos_short_file_name(file, filename_set):
    if False:
        print('Hello World!')
    ' See http://support.microsoft.com/default.aspx?scid=kb;en-us;Q142982\n\n    These are no complete 8.3 dos short names. The ~ char is missing and \n    replaced with one character from the filename. WiX warns about such\n    filenames, since a collision might occur. Google for "CNDL1014" for\n    more information.\n    '
    if is_dos_short_file_name(file):
        return file
    (fname, ext) = os.path.splitext(file)
    file = file.upper()
    if is_dos_short_file_name(file):
        return file
    forbidden = '."/[]:;=, '
    fname = ''.join([c for c in fname if c not in forbidden])
    (duplicate, num) = (not None, 1)
    while duplicate:
        shortname = '%s%s' % (fname[:8 - len(str(num))].upper(), str(num))
        if len(ext) >= 2:
            shortname = '%s%s' % (shortname, ext[:4].upper())
        (duplicate, num) = (shortname in filename_set, num + 1)
    assert is_dos_short_file_name(shortname), 'shortname is %s, longname is %s' % (shortname, file)
    filename_set.append(shortname)
    return shortname

def create_feature_dict(files):
    if False:
        print('Hello World!')
    " X_MSI_FEATURE and doc FileTag's can be used to collect files in a\n        hierarchy. This function collects the files into this hierarchy.\n    "
    dict = {}

    def add_to_dict(feature, file):
        if False:
            for i in range(10):
                print('nop')
        if not SCons.Util.is_List(feature):
            feature = [feature]
        for f in feature:
            if f not in dict:
                dict[f] = [file]
            else:
                dict[f].append(file)
    for file in files:
        if hasattr(file, 'PACKAGING_X_MSI_FEATURE'):
            add_to_dict(file.PACKAGING_X_MSI_FEATURE, file)
        elif hasattr(file, 'PACKAGING_DOC'):
            add_to_dict('PACKAGING_DOC', file)
        else:
            add_to_dict('default', file)
    return dict

def generate_guids(root):
    if False:
        for i in range(10):
            print('nop')
    ' generates globally unique identifiers for parts of the xml which need\n    them.\n\n    Component tags have a special requirement. Their UUID is only allowed to\n    change if the list of their contained resources has changed. This allows\n    for clean removal and proper updates.\n\n    To handle this requirement, the uuid is generated with an md5 hashing the\n    whole subtree of a xml node.\n    '
    import uuid
    needs_id = {'Product': 'Id', 'Package': 'Id', 'Component': 'Guid'}
    for (key, value) in needs_id.items():
        node_list = root.getElementsByTagName(key)
        attribute = value
        for node in node_list:
            hash = uuid.uuid5(uuid.NAMESPACE_URL, node.toxml())
            node.attributes[attribute] = str(hash)

def string_wxsfile(target, source, env):
    if False:
        i = 10
        return i + 15
    return 'building WiX file %s' % target[0].path

def build_wxsfile(target, source, env):
    if False:
        for i in range(10):
            print('nop')
    " Compiles a .wxs file from the keywords given in env['msi_spec'] and\n        by analyzing the tree of source nodes and their tags.\n    "
    f = open(target[0].get_abspath(), 'w')
    try:
        doc = Document()
        root = doc.createElement('Wix')
        root.attributes['xmlns'] = 'http://schemas.microsoft.com/wix/2003/01/wi'
        doc.appendChild(root)
        filename_set = []
        id_set = {}
        build_wxsfile_header_section(root, env)
        build_wxsfile_file_section(root, source, env['NAME'], env['VERSION'], env['VENDOR'], filename_set, id_set)
        generate_guids(root)
        build_wxsfile_features_section(root, source, env['NAME'], env['VERSION'], env['SUMMARY'], id_set)
        build_wxsfile_default_gui(root)
        build_license_file(target[0].get_dir(), env)
        f.write(doc.toprettyxml())
        if 'CHANGE_SPECFILE' in env:
            env['CHANGE_SPECFILE'](target, source)
    except KeyError as e:
        raise SCons.Errors.UserError('"%s" package field for MSI is missing.' % e.args[0])
    finally:
        f.close()

def create_default_directory_layout(root, NAME, VERSION, VENDOR, filename_set):
    if False:
        while True:
            i = 10
    " Create the wix default target directory layout and return the innermost\n    directory.\n\n    We assume that the XML tree delivered in the root argument already contains\n    the Product tag.\n\n    Everything is put under the PFiles directory property defined by WiX.\n    After that a directory  with the 'VENDOR' tag is placed and then a\n    directory with the name of the project and its VERSION. This leads to the\n    following TARGET Directory Layout:\n    C:\\<PFiles>\\<Vendor>\\<Projectname-Version>\\\n    Example: C:\\Programme\\Company\\Product-1.2\\\n    "
    doc = Document()
    d1 = doc.createElement('Directory')
    d1.attributes['Id'] = 'TARGETDIR'
    d1.attributes['Name'] = 'SourceDir'
    d2 = doc.createElement('Directory')
    d2.attributes['Id'] = 'ProgramFilesFolder'
    d2.attributes['Name'] = 'PFiles'
    d3 = doc.createElement('Directory')
    d3.attributes['Id'] = 'VENDOR_folder'
    d3.attributes['Name'] = escape(gen_dos_short_file_name(VENDOR, filename_set))
    d3.attributes['LongName'] = escape(VENDOR)
    d4 = doc.createElement('Directory')
    project_folder = '%s-%s' % (NAME, VERSION)
    d4.attributes['Id'] = 'MY_DEFAULT_FOLDER'
    d4.attributes['Name'] = escape(gen_dos_short_file_name(project_folder, filename_set))
    d4.attributes['LongName'] = escape(project_folder)
    d1.childNodes.append(d2)
    d2.childNodes.append(d3)
    d3.childNodes.append(d4)
    root.getElementsByTagName('Product')[0].childNodes.append(d1)
    return d4

def build_wxsfile_file_section(root, files, NAME, VERSION, VENDOR, filename_set, id_set):
    if False:
        while True:
            i = 10
    " Builds the Component sections of the wxs file with their included files.\n\n    Files need to be specified in 8.3 format and in the long name format, long\n    filenames will be converted automatically.\n\n    Features are specficied with the 'X_MSI_FEATURE' or 'DOC' FileTag.\n    "
    root = create_default_directory_layout(root, NAME, VERSION, VENDOR, filename_set)
    components = create_feature_dict(files)
    factory = Document()

    def get_directory(node, dir):
        if False:
            i = 10
            return i + 15
        ' Returns the node under the given node representing the directory.\n\n        Returns the component node if dir is None or empty.\n        '
        if dir == '' or not dir:
            return node
        Directory = node
        dir_parts = dir.split(os.path.sep)
        upper_dir = ''
        dir_parts = [d for d in dir_parts if d != '']
        for d in dir_parts[:]:
            already_created = [c for c in Directory.childNodes if c.nodeName == 'Directory' and c.attributes['LongName'].value == escape(d)]
            if already_created:
                Directory = already_created[0]
                dir_parts.remove(d)
                upper_dir += d
            else:
                break
        for d in dir_parts:
            nDirectory = factory.createElement('Directory')
            nDirectory.attributes['LongName'] = escape(d)
            nDirectory.attributes['Name'] = escape(gen_dos_short_file_name(d, filename_set))
            upper_dir += d
            nDirectory.attributes['Id'] = convert_to_id(upper_dir, id_set)
            Directory.childNodes.append(nDirectory)
            Directory = nDirectory
        return Directory
    for file in files:
        (drive, path) = os.path.splitdrive(file.PACKAGING_INSTALL_LOCATION)
        filename = os.path.basename(path)
        dirname = os.path.dirname(path)
        h = {'PACKAGING_X_MSI_VITAL': 'yes', 'PACKAGING_X_MSI_FILEID': convert_to_id(filename, id_set), 'PACKAGING_X_MSI_LONGNAME': filename, 'PACKAGING_X_MSI_SHORTNAME': gen_dos_short_file_name(filename, filename_set), 'PACKAGING_X_MSI_SOURCE': file.get_path()}
        for (k, v) in [(k, v) for (k, v) in h.items() if not hasattr(file, k)]:
            setattr(file, k, v)
        File = factory.createElement('File')
        File.attributes['LongName'] = escape(file.PACKAGING_X_MSI_LONGNAME)
        File.attributes['Name'] = escape(file.PACKAGING_X_MSI_SHORTNAME)
        File.attributes['Source'] = escape(file.PACKAGING_X_MSI_SOURCE)
        File.attributes['Id'] = escape(file.PACKAGING_X_MSI_FILEID)
        File.attributes['Vital'] = escape(file.PACKAGING_X_MSI_VITAL)
        Component = factory.createElement('Component')
        Component.attributes['DiskId'] = '1'
        Component.attributes['Id'] = convert_to_id(filename, id_set)
        Directory = get_directory(root, dirname)
        Directory.childNodes.append(Component)
        Component.childNodes.append(File)

def build_wxsfile_features_section(root, files, NAME, VERSION, SUMMARY, id_set):
    if False:
        return 10
    ' This function creates the <features> tag based on the supplied xml tree.\n\n    This is achieved by finding all <component>s and adding them to a default target.\n\n    It should be called after the tree has been built completly.  We assume\n    that a MY_DEFAULT_FOLDER Property is defined in the wxs file tree.\n\n    Furthermore a top-level with the name and VERSION of the software will be created.\n\n    An PACKAGING_X_MSI_FEATURE can either be a string, where the feature\n    DESCRIPTION will be the same as its title or a Tuple, where the first\n    part will be its title and the second its DESCRIPTION.\n    '
    factory = Document()
    Feature = factory.createElement('Feature')
    Feature.attributes['Id'] = 'complete'
    Feature.attributes['ConfigurableDirectory'] = 'MY_DEFAULT_FOLDER'
    Feature.attributes['Level'] = '1'
    Feature.attributes['Title'] = escape('%s %s' % (NAME, VERSION))
    Feature.attributes['Description'] = escape(SUMMARY)
    Feature.attributes['Display'] = 'expand'
    for (feature, files) in create_feature_dict(files).items():
        SubFeature = factory.createElement('Feature')
        SubFeature.attributes['Level'] = '1'
        if SCons.Util.is_Tuple(feature):
            SubFeature.attributes['Id'] = convert_to_id(feature[0], id_set)
            SubFeature.attributes['Title'] = escape(feature[0])
            SubFeature.attributes['Description'] = escape(feature[1])
        else:
            SubFeature.attributes['Id'] = convert_to_id(feature, id_set)
            if feature == 'default':
                SubFeature.attributes['Description'] = 'Main Part'
                SubFeature.attributes['Title'] = 'Main Part'
            elif feature == 'PACKAGING_DOC':
                SubFeature.attributes['Description'] = 'Documentation'
                SubFeature.attributes['Title'] = 'Documentation'
            else:
                SubFeature.attributes['Description'] = escape(feature)
                SubFeature.attributes['Title'] = escape(feature)
        for f in files:
            ComponentRef = factory.createElement('ComponentRef')
            ComponentRef.attributes['Id'] = convert_to_id(os.path.basename(f.get_path()), id_set)
            SubFeature.childNodes.append(ComponentRef)
        Feature.childNodes.append(SubFeature)
    root.getElementsByTagName('Product')[0].childNodes.append(Feature)

def build_wxsfile_default_gui(root):
    if False:
        while True:
            i = 10
    ' This function adds a default GUI to the wxs file\n    '
    factory = Document()
    Product = root.getElementsByTagName('Product')[0]
    UIRef = factory.createElement('UIRef')
    UIRef.attributes['Id'] = 'WixUI_Mondo'
    Product.childNodes.append(UIRef)
    UIRef = factory.createElement('UIRef')
    UIRef.attributes['Id'] = 'WixUI_ErrorProgressText'
    Product.childNodes.append(UIRef)

def build_license_file(directory, spec):
    if False:
        i = 10
        return i + 15
    ' Creates a License.rtf file with the content of "X_MSI_LICENSE_TEXT"\n    in the given directory\n    '
    (name, text) = ('', '')
    try:
        name = spec['LICENSE']
        text = spec['X_MSI_LICENSE_TEXT']
    except KeyError:
        pass
    if name != '' or text != '':
        with open(os.path.join(directory.get_path(), 'License.rtf'), 'w') as f:
            f.write('{\\rtf')
            if text != '':
                f.write(text.replace('\n', '\\par '))
            else:
                f.write(name + '\\par\\par')
            f.write('}')

def build_wxsfile_header_section(root, spec):
    if False:
        return 10
    ' Adds the xml file node which define the package meta-data.\n    '
    factory = Document()
    Product = factory.createElement('Product')
    Package = factory.createElement('Package')
    root.childNodes.append(Product)
    Product.childNodes.append(Package)
    if 'X_MSI_LANGUAGE' not in spec:
        spec['X_MSI_LANGUAGE'] = '1033'
    Product.attributes['Name'] = escape(spec['NAME'])
    Product.attributes['Version'] = escape(spec['VERSION'])
    Product.attributes['Manufacturer'] = escape(spec['VENDOR'])
    Product.attributes['Language'] = escape(spec['X_MSI_LANGUAGE'])
    Package.attributes['Description'] = escape(spec['SUMMARY'])
    if 'DESCRIPTION' in spec:
        Package.attributes['Comments'] = escape(spec['DESCRIPTION'])
    if 'X_MSI_UPGRADE_CODE' in spec:
        Package.attributes['X_MSI_UPGRADE_CODE'] = escape(spec['X_MSI_UPGRADE_CODE'])
    Media = factory.createElement('Media')
    Media.attributes['Id'] = '1'
    Media.attributes['Cabinet'] = 'default.cab'
    Media.attributes['EmbedCab'] = 'yes'
    root.getElementsByTagName('Product')[0].childNodes.append(Media)
wxs_builder = Builder(action=Action(build_wxsfile, string_wxsfile), ensure_suffix='.wxs')

def package(env, target, source, PACKAGEROOT, NAME, VERSION, DESCRIPTION, SUMMARY, VENDOR, X_MSI_LANGUAGE, **kw):
    if False:
        return 10
    SCons.Tool.Tool('wix').generate(env)
    loc = locals()
    del loc['kw']
    kw.update(loc)
    del kw['source'], kw['target'], kw['env']
    (target, source) = stripinstallbuilder(target, source, env)
    env['msi_spec'] = kw
    specfile = wxs_builder(*[env, target, source], **kw)
    msifile = env.WiX(target, specfile)
    return (msifile, source + [specfile])