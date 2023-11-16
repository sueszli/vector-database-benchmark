import io
import os
import re
import struct
import pathlib
from PyInstaller import log as logging
from PyInstaller.archive.writers import SplashWriter
from PyInstaller.building import splash_templates
from PyInstaller.building.datastruct import Target
from PyInstaller.building.utils import _check_guts_eq, _check_guts_toc, misc
from PyInstaller.compat import is_darwin
from PyInstaller.depend import bindepend
from PyInstaller.utils.hooks import tcl_tk as tcltk_utils
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None
logger = logging.getLogger(__name__)
splash_requirements = [os.path.join(tcltk_utils.TK_ROOTNAME, 'license.terms'), os.path.join(tcltk_utils.TK_ROOTNAME, 'text.tcl'), os.path.join(tcltk_utils.TK_ROOTNAME, 'tk.tcl'), os.path.join(tcltk_utils.TK_ROOTNAME, 'ttk', 'ttk.tcl'), os.path.join(tcltk_utils.TK_ROOTNAME, 'ttk', 'fonts.tcl'), os.path.join(tcltk_utils.TK_ROOTNAME, 'ttk', 'cursors.tcl'), os.path.join(tcltk_utils.TK_ROOTNAME, 'ttk', 'utils.tcl')]

class Splash(Target):
    """
    Bundles the required resources for the splash screen into a file, which will be included in the CArchive.

    A Splash has two outputs, one is itself and one is stored in splash.binaries. Both need to be passed to other
    build targets in order to enable the splash screen.
    """

    def __init__(self, image_file, binaries, datas, **kwargs):
        if False:
            while True:
                i = 10
        '\n        :param str image_file:\n            A path-like object to the image to be used. Only the PNG file format is supported.\n\n            .. note:: If a different file format is supplied and PIL (Pillow) is installed, the file will be converted\n                automatically.\n\n            .. note:: *Windows*: The color ``\'magenta\'`` / ``\'#ff00ff\'`` must not be used in the image or text, as it is\n                used by splash screen to indicate transparent areas. Use a similar color (e.g., ``\'#ff00fe\'``) instead.\n\n            .. note:: If PIL (Pillow) is installed and the image is bigger than max_img_size, the image will be resized\n                to fit into the specified area.\n        :param list binaries:\n            The TOC list of binaries the Analysis build target found. This TOC includes all extension modules and their\n            binary dependencies. This is required to determine whether the user\'s program uses `tkinter`.\n        :param list datas:\n            The TOC list of data the Analysis build target found. This TOC includes all data-file dependencies of the\n            modules. This is required to check if all splash screen requirements can be bundled.\n\n        :keyword text_pos:\n            An optional two-integer tuple that represents the origin of the text on the splash screen image. The\n            origin of the text is its lower left corner. A unit in the respective coordinate system is a pixel of the\n            image, its origin lies in the top left corner of the image. This parameter also acts like a switch for\n            the text feature. If omitted, no text will be displayed on the splash screen. This text will be used to\n            show textual progress in onefile mode.\n        :type text_pos: Tuple[int, int]\n        :keyword text_size:\n            The desired size of the font. If the size argument is a positive number, it is interpreted as a size in\n            points. If size is a negative number, its absolute value is interpreted as a size in pixels. Default: ``12``\n        :type text_size: int\n        :keyword text_font:\n            An optional name of a font for the text. This font must be installed on the user system, otherwise the\n            system default font is used. If this parameter is omitted, the default font is also used.\n        :keyword text_color:\n            An optional color for the text. HTML color codes (``\'#40e0d0\'``) and color names (``\'turquoise\'``) are\n            supported. Default: ``\'black\'``\n            (Windows: the color ``\'magenta\'`` / ``\'#ff00ff\'`` is used to indicate transparency, and should not be used)\n        :type text_color: str\n        :keyword text_default:\n            The default text which will be displayed before the extraction starts. Default: ``"Initializing"``\n        :type text_default: str\n        :keyword full_tk:\n            By default Splash bundles only the necessary files for the splash screen (some tk components). This\n            options enables adding full tk and making it a requirement, meaning all tk files will be unpacked before\n            the splash screen can be started. This is useful during development of the splash screen script.\n            Default: ``False``\n        :type full_tk: bool\n        :keyword minify_script:\n            The splash screen is created by executing an Tcl/Tk script. This option enables minimizing the script,\n            meaning removing all non essential parts from the script. Default: ``True``\n        :keyword rundir:\n            The folder name in which tcl/tk will be extracted at runtime. There should be no matching folder in your\n            application to avoid conflicts. Default:  ``\'__splash\'``\n        :type rundir: str\n        :keyword name:\n            An optional alternative filename for the .res file. If not specified, a name is generated.\n        :type name: str\n        :keyword script_name:\n            An optional alternative filename for the Tcl script, that will be generated. If not specified, a name is\n            generated.\n        :type script_name: str\n        :keyword max_img_size:\n            Maximum size of the splash screen image as a tuple. If the supplied image exceeds this limit, it will be\n            resized to fit the maximum width (to keep the original aspect ratio). This option can be disabled by\n            setting it to None. Default: ``(760, 480)``\n        :type max_img_size: Tuple[int, int]\n        :keyword always_on_top:\n            Force the splashscreen to be always on top of other windows. If disabled, other windows (e.g., from other\n            applications) can cover the splash screen by user bringing them to front. This might be useful for\n            frozen applications with long startup times. Default: ``True``\n        :type always_on_top: bool\n        '
        from ..config import CONF
        Target.__init__(self)
        if is_darwin:
            raise SystemExit('Splash screen is not supported on macOS.')
        if not os.path.isabs(image_file):
            image_file = os.path.join(CONF['specpath'], image_file)
        image_file = os.path.normpath(image_file)
        if not os.path.exists(image_file):
            raise ValueError("Image file '%s' not found" % image_file)
        self.image_file = image_file
        self.full_tk = kwargs.get('full_tk', False)
        self.name = kwargs.get('name', None)
        self.script_name = kwargs.get('script_name', None)
        self.minify_script = kwargs.get('minify_script', True)
        self.rundir = kwargs.get('rundir', None)
        self.max_img_size = kwargs.get('max_img_size', (760, 480))
        self.text_pos = kwargs.get('text_pos', None)
        self.text_size = kwargs.get('text_size', 12)
        self.text_font = kwargs.get('text_font', 'TkDefaultFont')
        self.text_color = kwargs.get('text_color', 'black')
        self.text_default = kwargs.get('text_default', 'Initializing')
        self.always_on_top = kwargs.get('always_on_top', True)
        root = os.path.splitext(self.tocfilename)[0]
        if self.name is None:
            self.name = root + '.res'
        if self.script_name is None:
            self.script_name = root + '_script.tcl'
        if self.rundir is None:
            self.rundir = self._find_rundir(binaries + datas)
        try:
            import _tkinter
            self._tkinter_module = _tkinter
            self._tkinter_file = self._tkinter_module.__file__
        except ModuleNotFoundError:
            raise SystemExit('Your platform does not support the splash screen feature, since tkinter is not installed. Please install tkinter and try again.')
        self.uses_tkinter = self._uses_tkinter(self._tkinter_file, binaries)
        logger.debug('Program uses tkinter: %r', self.uses_tkinter)
        self.script = self.generate_script()
        (self.tcl_lib, self.tk_lib) = tcltk_utils.find_tcl_tk_shared_libs(self._tkinter_file)
        if is_darwin:
            if self.tcl_lib[1] is None or 'Library/Frameworks/Tcl.framework' in self.tcl_lib[1]:
                raise SystemExit('The splash screen feature does not support macOS system framework version of Tcl/Tk.')
        assert all(self.tcl_lib)
        assert all(self.tk_lib)
        logger.debug('Use Tcl Library from %s and Tk From %s', self.tcl_lib, self.tk_lib)
        self.splash_requirements = set([self.tcl_lib[0], self.tk_lib[0]] + splash_requirements)
        logger.info('Collect tcl/tk binaries for the splash screen')
        tcltk_tree = tcltk_utils.collect_tcl_tk_files(self._tkinter_file)
        if self.full_tk:
            self.splash_requirements.update((entry[0] for entry in tcltk_tree))
        tcltk_libs = [(dest_name, src_name, 'BINARY') for (dest_name, src_name) in (self.tcl_lib, self.tk_lib)]
        self.binaries = bindepend.binary_dependency_analysis(tcltk_libs)
        self.splash_requirements.update((entry[0] for entry in self.binaries))
        if not self.uses_tkinter:
            self.binaries.extend((entry for entry in tcltk_tree if entry[0] in self.splash_requirements))
        collected_files = set((entry[0] for entry in binaries + datas + self.binaries))

        def _filter_requirement(filename):
            if False:
                print('Hello World!')
            if filename not in collected_files:
                logger.warning('The local Tcl/Tk installation is missing the file %s. The behavior of the splash screen is therefore undefined and may be unsupported.', filename)
                return False
            return True
        self.splash_requirements = set(filter(_filter_requirement, self.splash_requirements))
        self.test_tk_version()
        logger.debug('Splash Requirements: %s', self.splash_requirements)
        self.__postinit__()
    _GUTS = (('image_file', _check_guts_eq), ('name', _check_guts_eq), ('script_name', _check_guts_eq), ('text_pos', _check_guts_eq), ('text_size', _check_guts_eq), ('text_font', _check_guts_eq), ('text_color', _check_guts_eq), ('text_default', _check_guts_eq), ('always_on_top', _check_guts_eq), ('full_tk', _check_guts_eq), ('minify_script', _check_guts_eq), ('rundir', _check_guts_eq), ('max_img_size', _check_guts_eq), ('uses_tkinter', _check_guts_eq), ('script', _check_guts_eq), ('tcl_lib', _check_guts_eq), ('tk_lib', _check_guts_eq), ('splash_requirements', _check_guts_eq), ('binaries', _check_guts_toc), ('_tkinter_file', _check_guts_eq))

    def _check_guts(self, data, last_build):
        if False:
            print('Hello World!')
        if Target._check_guts(self, data, last_build):
            return True
        if misc.mtime(self.image_file) > last_build:
            logger.info('Building %s because file %s changed', self.tocbasename, self.image_file)
            return True
        return False

    def assemble(self):
        if False:
            while True:
                i = 10
        logger.info('Building Splash %s', self.name)

        def _resize_image(_image, _orig_size):
            if False:
                i = 10
                return i + 15
            if PILImage:
                (_w, _h) = _orig_size
                _ratio_w = self.max_img_size[0] / _w
                if _ratio_w < 1:
                    _h = int(_h * _ratio_w)
                    _w = self.max_img_size[0]
                _ratio_h = self.max_img_size[1] / _h
                if _ratio_h < 1:
                    _w = int(_w * _ratio_h)
                    _h = self.max_img_size[1]
                if isinstance(_image, PILImage.Image):
                    _img = _image
                else:
                    _img = PILImage.open(_image)
                _img_resized = _img.resize((_w, _h))
                _image_stream = io.BytesIO()
                _img_resized.save(_image_stream, format='PNG')
                _img.close()
                _img_resized.close()
                _image_data = _image_stream.getvalue()
                logger.info('Resized image %s from dimensions %s to (%d, %d)', self.image_file, str(_orig_size), _w, _h)
                return _image_data
            else:
                raise ValueError('The splash image dimensions (w: %d, h: %d) exceed max_img_size (w: %d, h:%d), but the image cannot be resized due to missing PIL.Image! Either install the Pillow package, adjust the max_img_size, or use an image of compatible dimensions.', _orig_size[0], _orig_size[1], self.max_img_size[0], self.max_img_size[1])
        image_file = open(self.image_file, 'rb')
        if image_file.read(8) == b'\x89PNG\r\n\x1a\n':
            image_file.seek(16)
            img_size = (struct.unpack('!I', image_file.read(4))[0], struct.unpack('!I', image_file.read(4))[0])
            if img_size > self.max_img_size:
                image = _resize_image(self.image_file, img_size)
            else:
                image = os.path.abspath(self.image_file)
        elif PILImage:
            img = PILImage.open(self.image_file, mode='r')
            if img.size > self.max_img_size:
                image = _resize_image(img, img.size)
            else:
                image_data = io.BytesIO()
                img.save(image_data, format='PNG')
                img.close()
                image = image_data.getvalue()
            logger.info('Converted image %s to PNG format', self.image_file)
        else:
            raise ValueError('The image %s needs to be converted to a PNG file, but PIL.Image is not available! Either install the Pillow package, or use a PNG image for you splash screen.', self.image_file)
        image_file.close()
        SplashWriter(self.name, self.splash_requirements, self.tcl_lib[0], self.tk_lib[0], tcltk_utils.TK_ROOTNAME, self.rundir, image, self.script)

    def test_tk_version(self):
        if False:
            print('Hello World!')
        tcl_version = float(self._tkinter_module.TCL_VERSION)
        tk_version = float(self._tkinter_module.TK_VERSION)
        if tcl_version < 8.6 or tk_version < 8.6:
            logger.warning('The installed Tcl/Tk (%s/%s) version might not work with the splash screen feature of the bootloader. The bootloader is tested against Tcl/Tk 8.6', self._tkinter_module.TCL_VERSION, self._tkinter_module.TK_VERSION)
        if tcl_version != tk_version:
            logger.warning('The installed version of Tcl (%s) and Tk (%s) do not match. PyInstaller is tested against matching versions', self._tkinter_module.TCL_VERSION, self._tkinter_module.TK_VERSION)
        if not tcltk_utils.tcl_threaded:
            raise SystemExit('The installed tcl version is not threaded. PyInstaller only supports the splash screen using threaded tcl.')

    def generate_script(self):
        if False:
            return 10
        '\n        Generate the script for the splash screen.\n\n        If minify_script is True, all unnecessary parts will be removed.\n        '
        d = {}
        if self.text_pos is not None:
            logger.debug('Add text support to splash screen')
            d.update({'pad_x': self.text_pos[0], 'pad_y': self.text_pos[1], 'color': self.text_color, 'font': self.text_font, 'font_size': self.text_size, 'default_text': self.text_default})
        script = splash_templates.build_script(text_options=d, always_on_top=self.always_on_top)
        if self.minify_script:
            script = '\n'.join((line for line in map(lambda line: line.strip(), script.splitlines()) if not line.startswith('#') and line))
            script = re.sub(' +', ' ', script)
        with open(self.script_name, 'w') as script_file:
            script_file.write(script)
        return script

    @staticmethod
    def _uses_tkinter(tkinter_file, binaries):
        if False:
            print('Hello World!')
        tkinter_file = pathlib.PurePath(tkinter_file)
        for (dest_name, src_name, typecode) in binaries:
            if pathlib.PurePath(src_name) == tkinter_file:
                return True
        return False

    @staticmethod
    def _find_rundir(structure):
        if False:
            while True:
                i = 10
        rundir = '__splash%s'
        candidate = rundir % ''
        counter = 0
        while any((e[0].startswith(candidate + os.sep) for e in structure)):
            candidate = rundir % str(counter)
            counter += 1
            assert len(candidate) <= 16
        return candidate