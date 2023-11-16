from base64 import b64decode
import os
import re
from warnings import warn
from qtpy import QtCore, QtGui, QtWidgets
from traitlets import Bool
from pygments.util import ClassNotFound
from qtconsole.svg import save_svg, svg_to_clipboard, svg_to_image
from .jupyter_widget import JupyterWidget
from .styles import get_colors
try:
    from IPython.lib.latextools import latex_to_png
except ImportError:
    latex_to_png = None

def _ensure_dir_exists(path, mode=493):
    if False:
        for i in range(10):
            print('nop')
    "ensure that a directory exists\n\n    If it doesn't exists, try to create it and protect against a race condition\n    if another process is doing the same.\n\n    The default permissions are 755, which differ from os.makedirs default of 777.\n    "
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError('%r exists but is not a directory' % path)

class LatexError(Exception):
    """Exception for Latex errors"""

class RichIPythonWidget(JupyterWidget):
    """Dummy class for config inheritance. Destroyed below."""

class RichJupyterWidget(RichIPythonWidget):
    """ An JupyterWidget that supports rich text, including lists, images, and
        tables. Note that raw performance will be reduced compared to the plain
        text version.
    """
    _payload_source_plot = 'ipykernel.pylab.backend_payload.add_plot_payload'
    _jpg_supported = Bool(False)
    _svg_warning_displayed = False

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        ' Create a RichJupyterWidget.\n        '
        kw['kind'] = 'rich'
        super().__init__(*args, **kw)
        self._html_exporter.image_tag = self._get_image_tag
        self._name_to_svg_map = {}
        self._jpg_supported = 'jpeg' in QtGui.QImageReader.supportedImageFormats()

    def export_html(self):
        if False:
            for i in range(10):
                print('nop')
        ' Shows a dialog to export HTML/XML in various formats.\n\n        Overridden in order to reset the _svg_warning_displayed flag prior\n        to the export running.\n        '
        self._svg_warning_displayed = False
        super().export_html()

    def _context_menu_make(self, pos):
        if False:
            print('Hello World!')
        ' Reimplemented to return a custom context menu for images.\n        '
        format = self._control.cursorForPosition(pos).charFormat()
        name = format.stringProperty(QtGui.QTextFormat.ImageName)
        if name:
            menu = QtWidgets.QMenu(self)
            menu.addAction('Copy Image', lambda : self._copy_image(name))
            menu.addAction('Save Image As...', lambda : self._save_image(name))
            menu.addSeparator()
            svg = self._name_to_svg_map.get(name, None)
            if svg is not None:
                menu.addSeparator()
                menu.addAction('Copy SVG', lambda : svg_to_clipboard(svg))
                menu.addAction('Save SVG As...', lambda : save_svg(svg, self._control))
        else:
            menu = super()._context_menu_make(pos)
        return menu

    def _pre_image_append(self, msg, prompt_number):
        if False:
            return 10
        'Append the Out[] prompt  and make the output nicer\n\n        Shared code for some the following if statement\n        '
        self._append_plain_text(self.output_sep, True)
        self._append_html(self._make_out_prompt(prompt_number), True)
        self._append_plain_text('\n', True)

    def _handle_execute_result(self, msg):
        if False:
            i = 10
            return i + 15
        'Overridden to handle rich data types, like SVG.'
        self.log.debug('execute_result: %s', msg.get('content', ''))
        if self.include_output(msg):
            self.flush_clearoutput()
            content = msg['content']
            prompt_number = content.get('execution_count', 0)
            data = content['data']
            metadata = msg['content']['metadata']
            if 'image/svg+xml' in data:
                self._pre_image_append(msg, prompt_number)
                self._append_svg(data['image/svg+xml'], True)
                self._append_html(self.output_sep2, True)
            elif 'image/png' in data:
                self._pre_image_append(msg, prompt_number)
                png = b64decode(data['image/png'].encode('ascii'))
                self._append_png(png, True, metadata=metadata.get('image/png', None))
                self._append_html(self.output_sep2, True)
            elif 'image/jpeg' in data and self._jpg_supported:
                self._pre_image_append(msg, prompt_number)
                jpg = b64decode(data['image/jpeg'].encode('ascii'))
                self._append_jpg(jpg, True, metadata=metadata.get('image/jpeg', None))
                self._append_html(self.output_sep2, True)
            elif 'text/latex' in data:
                self._pre_image_append(msg, prompt_number)
                try:
                    self._append_latex(data['text/latex'], True)
                except LatexError:
                    return super()._handle_display_data(msg)
                self._append_html(self.output_sep2, True)
            else:
                return super()._handle_execute_result(msg)

    def _handle_display_data(self, msg):
        if False:
            i = 10
            return i + 15
        'Overridden to handle rich data types, like SVG.'
        self.log.debug('display_data: %s', msg.get('content', ''))
        if self.include_output(msg):
            self.flush_clearoutput()
            data = msg['content']['data']
            metadata = msg['content']['metadata']
            self.log.debug('display: %s', msg.get('content', ''))
            if 'image/svg+xml' in data:
                svg = data['image/svg+xml']
                self._append_svg(svg, True)
            elif 'image/png' in data:
                png = b64decode(data['image/png'].encode('ascii'))
                self._append_png(png, True, metadata=metadata.get('image/png', None))
            elif 'image/jpeg' in data and self._jpg_supported:
                jpg = b64decode(data['image/jpeg'].encode('ascii'))
                self._append_jpg(jpg, True, metadata=metadata.get('image/jpeg', None))
            elif 'text/latex' in data and latex_to_png:
                try:
                    self._append_latex(data['text/latex'], True)
                except LatexError:
                    return super()._handle_display_data(msg)
            else:
                return super()._handle_display_data(msg)

    def _is_latex_math(self, latex):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if a Latex string is in math mode\n\n        This is the only mode supported by qtconsole\n        '
        basic_envs = ['math', 'displaymath']
        starable_envs = ['equation', 'eqnarraymultline', 'gather', 'align', 'flalign', 'alignat']
        star_envs = [env + '*' for env in starable_envs]
        envs = basic_envs + starable_envs + star_envs
        env_syntax = ['\\begin{{{0}}} \\end{{{0}}}'.format(env).split() for env in envs]
        math_syntax = [('\\[', '\\]'), ('\\(', '\\)'), ('$$', '$$'), ('$', '$')]
        for (start, end) in math_syntax + env_syntax:
            inner = latex[len(start):-len(end)]
            if start in inner or end in inner:
                return False
            if latex.startswith(start) and latex.endswith(end):
                return True
        return False

    def _get_color(self, color):
        if False:
            while True:
                i = 10
        'Get color from the current syntax style if loadable.'
        try:
            return get_colors(self.syntax_style)[color]
        except ClassNotFound:
            return get_colors('default')[color]

    def _append_latex(self, latex, before_prompt=False, metadata=None):
        if False:
            print('Hello World!')
        ' Append latex data to the widget.'
        png = None
        if self._is_latex_math(latex):
            png = latex_to_png(latex, wrap=False, backend='dvipng', color=self._get_color('fgcolor'))
        if png is None and latex.startswith('$') and latex.endswith('$'):
            try:
                png = latex_to_png(latex, wrap=False, backend='matplotlib', color=self._get_color('fgcolor'))
            except Exception:
                pass
        if png:
            self._append_png(png, before_prompt, metadata)
        else:
            raise LatexError

    def _append_jpg(self, jpg, before_prompt=False, metadata=None):
        if False:
            i = 10
            return i + 15
        ' Append raw JPG data to the widget.'
        self._append_custom(self._insert_jpg, jpg, before_prompt, metadata=metadata)

    def _append_png(self, png, before_prompt=False, metadata=None):
        if False:
            return 10
        ' Append raw PNG data to the widget.\n        '
        self._append_custom(self._insert_png, png, before_prompt, metadata=metadata)

    def _append_svg(self, svg, before_prompt=False):
        if False:
            while True:
                i = 10
        ' Append raw SVG data to the widget.\n        '
        self._append_custom(self._insert_svg, svg, before_prompt)

    def _add_image(self, image):
        if False:
            return 10
        ' Adds the specified QImage to the document and returns a\n            QTextImageFormat that references it.\n        '
        document = self._control.document()
        name = str(image.cacheKey())
        document.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(name), image)
        format = QtGui.QTextImageFormat()
        format.setName(name)
        return format

    def _copy_image(self, name):
        if False:
            i = 10
            return i + 15
        " Copies the ImageResource with 'name' to the clipboard.\n        "
        image = self._get_image(name)
        QtWidgets.QApplication.clipboard().setImage(image)

    def _get_image(self, name):
        if False:
            print('Hello World!')
        " Returns the QImage stored as the ImageResource with 'name'.\n        "
        document = self._control.document()
        image = document.resource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(name))
        return image

    def _get_image_tag(self, match, path=None, format='png'):
        if False:
            print('Hello World!')
        ' Return (X)HTML mark-up for the image-tag given by match.\n\n        Parameters\n        ----------\n        match : re.SRE_Match\n            A match to an HTML image tag as exported by Qt, with\n            match.group("Name") containing the matched image ID.\n\n        path : string|None, optional [default None]\n            If not None, specifies a path to which supporting files may be\n            written (e.g., for linked images).  If None, all images are to be\n            included inline.\n\n        format : "png"|"svg"|"jpg", optional [default "png"]\n            Format for returned or referenced images.\n        '
        if format in ('png', 'jpg'):
            try:
                image = self._get_image(match.group('name'))
            except KeyError:
                return "<b>Couldn't find image %s</b>" % match.group('name')
            if path is not None:
                _ensure_dir_exists(path)
                relpath = os.path.basename(path)
                if image.save('%s/qt_img%s.%s' % (path, match.group('name'), format), 'PNG'):
                    return '<img src="%s/qt_img%s.%s">' % (relpath, match.group('name'), format)
                else:
                    return "<b>Couldn't save image!</b>"
            else:
                ba = QtCore.QByteArray()
                buffer_ = QtCore.QBuffer(ba)
                buffer_.open(QtCore.QIODevice.WriteOnly)
                image.save(buffer_, format.upper())
                buffer_.close()
                return '<img src="data:image/%s;base64,\n%s\n" />' % (format, re.sub('(.{60})', '\\1\\n', str(ba.toBase64().data().decode())))
        elif format == 'svg':
            try:
                svg = str(self._name_to_svg_map[match.group('name')])
            except KeyError:
                if not self._svg_warning_displayed:
                    QtWidgets.QMessageBox.warning(self, 'Error converting PNG to SVG.', "Cannot convert PNG images to SVG, export with PNG figures instead. If you want to export matplotlib figures as SVG, add to your ipython config:\n\n\tc.InlineBackend.figure_format = 'svg'\n\nAnd regenerate the figures.", QtWidgets.QMessageBox.Ok)
                    self._svg_warning_displayed = True
                return "<b>Cannot convert  PNG images to SVG.</b>  You must export this session with PNG images. If you want to export matplotlib figures as SVG, add to your config <span>c.InlineBackend.figure_format = 'svg'</span> and regenerate the figures."
            offset = svg.find('<svg')
            assert offset > -1
            return svg[offset:]
        else:
            return '<b>Unrecognized image format</b>'

    def _insert_jpg(self, cursor, jpg, metadata=None):
        if False:
            return 10
        ' Insert raw PNG data into the widget.'
        self._insert_img(cursor, jpg, 'jpg', metadata=metadata)

    def _insert_png(self, cursor, png, metadata=None):
        if False:
            while True:
                i = 10
        ' Insert raw PNG data into the widget.\n        '
        self._insert_img(cursor, png, 'png', metadata=metadata)

    def _insert_img(self, cursor, img, fmt, metadata=None):
        if False:
            return 10
        ' insert a raw image, jpg or png '
        if metadata:
            width = metadata.get('width', None)
            height = metadata.get('height', None)
        else:
            width = height = None
        try:
            image = QtGui.QImage()
            image.loadFromData(img, fmt.upper())
            if width and height:
                image = image.scaled(int(width), int(height), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            elif width and (not height):
                image = image.scaledToWidth(int(width), QtCore.Qt.SmoothTransformation)
            elif height and (not width):
                image = image.scaledToHeight(int(height), QtCore.Qt.SmoothTransformation)
        except ValueError:
            self._insert_plain_text(cursor, 'Received invalid %s data.' % fmt)
        else:
            format = self._add_image(image)
            cursor.insertBlock()
            cursor.insertImage(format)
            cursor.insertBlock()

    def _insert_svg(self, cursor, svg):
        if False:
            for i in range(10):
                print('nop')
        ' Insert raw SVG data into the widet.\n        '
        try:
            image = svg_to_image(svg)
        except ValueError:
            self._insert_plain_text(cursor, 'Received invalid SVG data.')
        else:
            format = self._add_image(image)
            self._name_to_svg_map[format.name()] = svg
            cursor.insertBlock()
            cursor.insertImage(format)
            cursor.insertBlock()

    def _save_image(self, name, format='PNG'):
        if False:
            return 10
        " Shows a save dialog for the ImageResource with 'name'.\n        "
        dialog = QtWidgets.QFileDialog(self._control, 'Save Image')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setDefaultSuffix(format.lower())
        dialog.setNameFilter('%s file (*.%s)' % (format, format.lower()))
        if dialog.exec_():
            filename = dialog.selectedFiles()[0]
            image = self._get_image(name)
            image.save(filename, format)

class RichIPythonWidget(RichJupyterWidget):
    """Deprecated class. Use RichJupyterWidget."""

    def __init__(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        warn('RichIPythonWidget is deprecated, use RichJupyterWidget', DeprecationWarning)
        super().__init__(*a, **kw)