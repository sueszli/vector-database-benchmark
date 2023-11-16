"""Internal support module for Android builds."""
import xml.etree.ElementTree as ET
from ._proto.targeting_pb2 import Abi
from ._proto.config_pb2 import BundleConfig
from ._proto.files_pb2 import NativeLibraries
from ._proto.Resources_pb2 import ResourceTable
from ._proto.Resources_pb2 import XmlNode
AbiAlias = Abi.AbiAlias

def str_resource(id):
    if False:
        while True:
            i = 10

    def compile(attrib, manifest):
        if False:
            i = 10
            return i + 15
        attrib.resource_id = id
    return compile

def int_resource(id):
    if False:
        print('Hello World!')

    def compile(attrib, manifest):
        if False:
            while True:
                i = 10
        attrib.resource_id = id
        if attrib.value.startswith('0x') or attrib.value.startswith('0X'):
            attrib.compiled_item.prim.int_hexadecimal_value = int(attrib.value, 16)
        else:
            attrib.compiled_item.prim.int_decimal_value = int(attrib.value)
    return compile

def bool_resource(id):
    if False:
        return 10

    def compile(attrib, manifest):
        if False:
            return 10
        attrib.resource_id = id
        attrib.compiled_item.prim.boolean_value = {'true': True, '1': True, 'false': False, '0': False}[attrib.value]
    return compile

def enum_resource(id, *values):
    if False:
        print('Hello World!')

    def compile(attrib, manifest):
        if False:
            i = 10
            return i + 15
        attrib.resource_id = id
        attrib.compiled_item.prim.int_decimal_value = values.index(attrib.value)
    return compile

def flag_resource(id, **values):
    if False:
        return 10

    def compile(attrib, manifest):
        if False:
            for i in range(10):
                print('nop')
        attrib.resource_id = id
        bitmask = 0
        flags = attrib.value.split('|')
        for flag in flags:
            bitmask = values[flag]
        attrib.compiled_item.prim.int_hexadecimal_value = bitmask
    return compile

def ref_resource(id):
    if False:
        for i in range(10):
            print('nop')

    def compile(attrib, manifest):
        if False:
            print('Hello World!')
        assert attrib.value[0] == '@'
        (ref_type, ref_name) = attrib.value[1:].split('/')
        attrib.resource_id = id
        attrib.compiled_item.ref.name = ref_type + '/' + ref_name
        if ref_type == 'android:style':
            attrib.compiled_item.ref.id = ANDROID_STYLES[ref_name]
        elif ':' not in ref_type:
            attrib.compiled_item.ref.id = manifest.register_resource(ref_type, ref_name)
        else:
            print(f'Warning: unhandled AndroidManifest.xml reference "{attrib.value}"')
    return compile
ANDROID_STYLES = {'Animation': 16973824, 'Animation.Activity': 16973825, 'Animation.Dialog': 16973826, 'Animation.Translucent': 16973827, 'Animation.Toast': 16973828, 'Theme': 16973829, 'Theme.NoTitleBar': 16973830, 'Theme.NoTitleBar.Fullscreen': 16973831, 'Theme.Black': 16973832, 'Theme.Black.NoTitleBar': 16973833, 'Theme.Black.NoTitleBar.Fullscreen': 16973834, 'Theme.Dialog': 16973835, 'Theme.Light': 16973836, 'Theme.Light.NoTitleBar': 16973837, 'Theme.Light.NoTitleBar.Fullscreen': 16973838, 'Theme.Translucent': 16973839, 'Theme.Translucent.NoTitleBar': 16973840, 'Theme.Translucent.NoTitleBar.Fullscreen': 16973841, 'Widget': 16973842, 'Widget.AbsListView': 16973843, 'Widget.Button': 16973844, 'Widget.Button.Inset': 16973845, 'Widget.Button.Small': 16973846, 'Widget.Button.Toggle': 16973847, 'Widget.CompoundButton': 16973848, 'Widget.CompoundButton.CheckBox': 16973849, 'Widget.CompoundButton.RadioButton': 16973850, 'Widget.CompoundButton.Star': 16973851, 'Widget.ProgressBar': 16973852, 'Widget.ProgressBar.Large': 16973853, 'Widget.ProgressBar.Small': 16973854, 'Widget.ProgressBar.Horizontal': 16973855, 'Widget.SeekBar': 16973856, 'Widget.RatingBar': 16973857, 'Widget.TextView': 16973858, 'Widget.EditText': 16973859, 'Widget.ExpandableListView': 16973860, 'Widget.ImageWell': 16973861, 'Widget.ImageButton': 16973862, 'Widget.AutoCompleteTextView': 16973863, 'Widget.Spinner': 16973864, 'Widget.TextView.PopupMenu': 16973865, 'Widget.TextView.SpinnerItem': 16973866, 'Widget.DropDownItem': 16973867, 'Widget.DropDownItem.Spinner': 16973868, 'Widget.ScrollView': 16973869, 'Widget.ListView': 16973870, 'Widget.ListView.White': 16973871, 'Widget.ListView.DropDown': 16973872, 'Widget.ListView.Menu': 16973873, 'Widget.GridView': 16973874, 'Widget.WebView': 16973875, 'Widget.TabWidget': 16973876, 'Widget.Gallery': 16973877, 'Widget.PopupWindow': 16973878, 'MediaButton': 16973879, 'MediaButton.Previous': 16973880, 'MediaButton.Next': 16973881, 'MediaButton.Play': 16973882, 'MediaButton.Ffwd': 16973883, 'MediaButton.Rew': 16973884, 'MediaButton.Pause': 16973885, 'TextAppearance': 16973886, 'TextAppearance.Inverse': 16973887, 'TextAppearance.Theme': 16973888, 'TextAppearance.DialogWindowTitle': 16973889, 'TextAppearance.Large': 16973890, 'TextAppearance.Large.Inverse': 16973891, 'TextAppearance.Medium': 16973892, 'TextAppearance.Medium.Inverse': 16973893, 'TextAppearance.Small': 16973894, 'TextAppearance.Small.Inverse': 16973895, 'TextAppearance.Theme.Dialog': 16973896, 'TextAppearance.Widget': 16973897, 'TextAppearance.Widget.Button': 16973898, 'TextAppearance.Widget.IconMenu.Item': 16973899, 'TextAppearance.Widget.EditText': 16973900, 'TextAppearance.Widget.TabWidget': 16973901, 'TextAppearance.Widget.TextView': 16973902, 'TextAppearance.Widget.TextView.PopupMenu': 16973903, 'TextAppearance.Widget.DropDownHint': 16973904, 'TextAppearance.Widget.DropDownItem': 16973905, 'TextAppearance.Widget.TextView.SpinnerItem': 16973906, 'TextAppearance.WindowTitle': 16973907}
ANDROID_ATTRIBUTES = {'allowBackup': bool_resource(16843392), 'allowClearUserData': bool_resource(16842757), 'allowParallelSyncs': bool_resource(16843570), 'allowSingleTap': bool_resource(16843353), 'allowTaskReparenting': bool_resource(16843268), 'alwaysRetainTaskState': bool_resource(16843267), 'clearTaskOnLaunch': bool_resource(16842773), 'debuggable': bool_resource(16842767), 'documentLaunchMode': enum_resource(16843845, 'none', 'intoExisting', 'always', 'never'), 'configChanges': flag_resource(16842783, mcc=1, mnc=2, locale=4, touchscreen=8, keyboard=16, keyboardHidden=32, navigation=64, orientation=128, screenLayout=256, uiMode=512, screenSize=1024, smallestScreenSize=2048, layoutDirection=8192, fontScale=1073741824), 'enabled': bool_resource(16842766), 'excludeFromRecents': bool_resource(16842775), 'exported': bool_resource(16842768), 'extractNativeLibs': bool_resource(16844010), 'finishOnTaskLaunch': bool_resource(16842772), 'fullBackupContent': bool_resource(16844011), 'glEsVersion': int_resource(16843393), 'hasCode': bool_resource(16842764), 'host': str_resource(16842792), 'icon': ref_resource(16842754), 'immersive': bool_resource(16843456), 'installLocation': enum_resource(16843447, 'auto', 'internalOnly', 'preferExternal'), 'isGame': bool_resource(16843764), 'label': str_resource(16842753), 'launchMode': enum_resource(16842781, 'standard', 'singleTop', 'singleTask', 'singleInstance'), 'maxSdkVersion': int_resource(16843377), 'mimeType': str_resource(16842790), 'minSdkVersion': int_resource(16843276), 'multiprocess': bool_resource(16842771), 'name': str_resource(16842755), 'noHistory': bool_resource(16843309), 'pathPattern': str_resource(16842796), 'resizeableActivity': bool_resource(16844022), 'required': bool_resource(16843406), 'scheme': str_resource(16842791), 'screenOrientation': enum_resource(16842782, 'landscape', 'portrait', 'user', 'behind', 'sensor', 'nosensor', 'sensorLandscape', 'sensorPortrait', 'reverseLandscape', 'reversePortrait', 'fullSensor', 'userLandscape', 'userPortrait', 'fullUser', 'locked'), 'stateNotNeeded': bool_resource(16842774), 'supportsRtl': bool_resource(16843695), 'supportsUploading': bool_resource(16843419), 'targetSandboxVersion': int_resource(16844108), 'targetSdkVersion': int_resource(16843376), 'theme': ref_resource(16842752), 'value': str_resource(16842788), 'versionCode': int_resource(16843291), 'versionName': str_resource(16843292)}

class AndroidManifest:

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self._stack = []
        self.root = XmlNode()
        self.resource_types = []
        self.resources = {}

    def parse_xml(self, data):
        if False:
            print('Hello World!')
        parser = ET.XMLParser(target=self)
        parser.feed(data)
        parser.close()

    def start_ns(self, prefix, uri):
        if False:
            while True:
                i = 10
        decl = self.root.element.namespace_declaration.add()
        decl.prefix = prefix
        decl.uri = uri

    def start(self, tag, attribs):
        if False:
            return 10
        if not self._stack:
            node = self.root
        else:
            node = self._stack[-1].child.add()
        element = node.element
        element.name = tag
        self._stack.append(element)
        for (key, value) in attribs.items():
            attrib = element.attribute.add()
            attrib.value = value
            if key.startswith('{'):
                (attrib.namespace_uri, key) = key[1:].split('}', 1)
                res_compile = ANDROID_ATTRIBUTES.get(key, None)
                if not res_compile:
                    print(f'Warning: unhandled AndroidManifest.xml attribute "{key}"')
            else:
                res_compile = None
            attrib.name = key
            if res_compile:
                res_compile(attrib, self)

    def end(self, tag):
        if False:
            print('Hello World!')
        self._stack.pop()

    def register_resource(self, type, name):
        if False:
            while True:
                i = 10
        if type not in self.resource_types:
            self.resource_types.append(type)
            type_id = len(self.resource_types)
            self.resources[type] = []
        else:
            type_id = self.resource_types.index(type) + 1
        resources = self.resources[type]
        if name in resources:
            entry_id = resources.index(name)
        else:
            entry_id = len(resources)
            resources.append(name)
        id = 127 << 24 | type_id << 16 | entry_id
        return id

    def dumps(self):
        if False:
            print('Hello World!')
        return self.root.SerializeToString()