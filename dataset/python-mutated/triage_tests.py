"""
This is a developer utility to help analyze and triage image
comparison failures.

It allows the failures to be quickly compared against the expected
results, and the new results to be either accepted (by copying the new
results to the source tree) or rejected (by copying the original
expected result to the source tree).

To start:

    If you ran the tests from the top-level of a source checkout, simply run:

        python tools/triage_tests.py

    Otherwise, you can manually select the location of `result_images`
    on the commandline.

Keys:

    left/right: Move between test, expected and diff images
    up/down:    Move between tests
    A:          Accept test.  Copy the test result to the source tree.
    R:          Reject test.  Copy the expected result to the source tree.
"""
import os
from pathlib import Path
import shutil
import sys
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.qt_compat import _exec
BASELINE_IMAGES = [Path('lib/matplotlib/tests/baseline_images'), *Path('lib/mpl_toolkits').glob('*/tests/baseline_images')]
exts = ['pdf', 'svg', 'eps']

class Thumbnail(QtWidgets.QFrame):
    """
    Represents one of the three thumbnails at the top of the window.
    """

    def __init__(self, parent, index, name):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.parent = parent
        self.index = index
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label, 0)
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.image.setMinimumSize(800 // 3, 600 // 3)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.parent.set_large_image(self.index)

class EventFilter(QtCore.QObject):

    def __init__(self, window):
        if False:
            print('Hello World!')
        super().__init__()
        self.window = window

    def eventFilter(self, receiver, event):
        if False:
            i = 10
            return i + 15
        if event.type() == QtCore.QEvent.Type.KeyPress:
            self.window.keyPressEvent(event)
            return True
        else:
            return super().eventFilter(receiver, event)

class Dialog(QtWidgets.QDialog):
    """
    The main dialog window.
    """

    def __init__(self, entries):
        if False:
            while True:
                i = 10
        super().__init__()
        self.entries = entries
        self.current_entry = -1
        self.current_thumbnail = -1
        event_filter = EventFilter(self)
        self.installEventFilter(event_filter)
        self.filelist = QtWidgets.QListWidget()
        self.filelist.setMinimumWidth(400)
        for entry in entries:
            self.filelist.addItem(entry.display)
        self.filelist.currentRowChanged.connect(self.set_entry)
        thumbnails_box = QtWidgets.QWidget()
        thumbnails_layout = QtWidgets.QVBoxLayout()
        self.thumbnails = []
        for (i, name) in enumerate(('test', 'expected', 'diff')):
            thumbnail = Thumbnail(self, i, name)
            thumbnails_layout.addWidget(thumbnail)
            self.thumbnails.append(thumbnail)
        thumbnails_box.setLayout(thumbnails_layout)
        images_layout = QtWidgets.QVBoxLayout()
        images_box = QtWidgets.QWidget()
        self.image_display = QtWidgets.QLabel()
        self.image_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.image_display.setMinimumSize(800, 600)
        images_layout.addWidget(self.image_display, 6)
        images_box.setLayout(images_layout)
        buttons_box = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout()
        accept_button = QtWidgets.QPushButton('Accept (A)')
        accept_button.clicked.connect(self.accept_test)
        buttons_layout.addWidget(accept_button)
        reject_button = QtWidgets.QPushButton('Reject (R)')
        reject_button.clicked.connect(self.reject_test)
        buttons_layout.addWidget(reject_button)
        buttons_box.setLayout(buttons_layout)
        images_layout.addWidget(buttons_box)
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.filelist, 1)
        main_layout.addWidget(thumbnails_box, 1)
        main_layout.addWidget(images_box, 3)
        self.setLayout(main_layout)
        self.setWindowTitle('matplotlib test triager')
        self.set_entry(0)

    def set_entry(self, index):
        if False:
            while True:
                i = 10
        if self.current_entry == index:
            return
        self.current_entry = index
        entry = self.entries[index]
        self.pixmaps = []
        for (fname, thumbnail) in zip(entry.thumbnails, self.thumbnails):
            pixmap = QtGui.QPixmap(os.fspath(fname))
            scaled_pixmap = pixmap.scaled(thumbnail.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            thumbnail.image.setPixmap(scaled_pixmap)
            self.pixmaps.append(scaled_pixmap)
        self.set_large_image(0)
        self.filelist.setCurrentRow(self.current_entry)

    def set_large_image(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.thumbnails[self.current_thumbnail].setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.current_thumbnail = index
        pixmap = QtGui.QPixmap(os.fspath(self.entries[self.current_entry].thumbnails[self.current_thumbnail]))
        self.image_display.setPixmap(pixmap)
        self.thumbnails[self.current_thumbnail].setFrameShape(QtWidgets.QFrame.Shape.Box)

    def accept_test(self):
        if False:
            print('Hello World!')
        entry = self.entries[self.current_entry]
        if entry.status == 'autogen':
            print('Cannot accept autogenerated test cases.')
            return
        entry.accept()
        self.filelist.currentItem().setText(self.entries[self.current_entry].display)
        self.set_entry(min(self.current_entry + 1, len(self.entries) - 1))

    def reject_test(self):
        if False:
            return 10
        entry = self.entries[self.current_entry]
        if entry.status == 'autogen':
            print('Cannot reject autogenerated test cases.')
            return
        entry.reject()
        self.filelist.currentItem().setText(self.entries[self.current_entry].display)
        self.set_entry(min(self.current_entry + 1, len(self.entries) - 1))

    def keyPressEvent(self, e):
        if False:
            return 10
        if e.key() == QtCore.Qt.Key.Key_Left:
            self.set_large_image((self.current_thumbnail - 1) % 3)
        elif e.key() == QtCore.Qt.Key.Key_Right:
            self.set_large_image((self.current_thumbnail + 1) % 3)
        elif e.key() == QtCore.Qt.Key.Key_Up:
            self.set_entry(max(self.current_entry - 1, 0))
        elif e.key() == QtCore.Qt.Key.Key_Down:
            self.set_entry(min(self.current_entry + 1, len(self.entries) - 1))
        elif e.key() == QtCore.Qt.Key.Key_A:
            self.accept_test()
        elif e.key() == QtCore.Qt.Key.Key_R:
            self.reject_test()
        else:
            super().keyPressEvent(e)

class Entry:
    """
    A model for a single image comparison test.
    """

    def __init__(self, path, root, source):
        if False:
            while True:
                i = 10
        self.source = source
        self.root = root
        self.dir = path.parent
        self.diff = path.name
        self.reldir = self.dir.relative_to(self.root)
        basename = self.diff[:-len('-failed-diff.png')]
        for ext in exts:
            if basename.endswith(f'_{ext}'):
                display_extension = f'_{ext}'
                extension = ext
                basename = basename[:-4]
                break
        else:
            display_extension = ''
            extension = 'png'
        self.basename = basename
        self.extension = extension
        self.generated = f'{basename}.{extension}'
        self.expected = f'{basename}-expected.{extension}'
        self.expected_display = f'{basename}-expected{display_extension}.png'
        self.generated_display = f'{basename}{display_extension}.png'
        self.name = self.reldir / self.basename
        self.destdir = self.get_dest_dir(self.reldir)
        self.thumbnails = [self.generated_display, self.expected_display, self.diff]
        self.thumbnails = [self.dir / x for x in self.thumbnails]
        if not Path(self.destdir, self.generated).exists():
            self.status = 'autogen'
        elif (self.dir / self.generated).read_bytes() == (self.destdir / self.generated).read_bytes():
            self.status = 'accept'
        else:
            self.status = 'unknown'

    def get_dest_dir(self, reldir):
        if False:
            return 10
        '\n        Find the source tree directory corresponding to the given\n        result_images subdirectory.\n        '
        for baseline_dir in BASELINE_IMAGES:
            path = self.source / baseline_dir / reldir
            if path.is_dir():
                return path
        raise ValueError(f"Can't find baseline dir for {reldir}")

    @property
    def display(self):
        if False:
            print('Hello World!')
        '\n        Get the display string for this entry.  This is the text that\n        appears in the list widget.\n        '
        status_map = {'unknown': '☐', 'accept': '☑', 'reject': '☒', 'autogen': '▣'}
        box = status_map[self.status]
        return f'{box} {self.name} [{self.extension}]'

    def accept(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Accept this test by copying the generated result to the source tree.\n        '
        copy_file(self.dir / self.generated, self.destdir / self.generated)
        self.status = 'accept'

    def reject(self):
        if False:
            print('Hello World!')
        '\n        Reject this test by copying the expected result to the source tree.\n        '
        expected = self.dir / self.expected
        if not expected.is_symlink():
            copy_file(expected, self.destdir / self.generated)
        self.status = 'reject'

def copy_file(a, b):
    if False:
        for i in range(10):
            print('nop')
    'Copy file from *a* to *b*.'
    print(f'copying: {a} to {b}')
    shutil.copyfile(a, b)

def find_failing_tests(result_images, source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find all of the failing tests by looking for files with\n    `-failed-diff` at the end of the basename.\n    '
    return [Entry(path, result_images, source) for path in sorted(Path(result_images).glob('**/*-failed-diff.*'))]

def launch(result_images, source):
    if False:
        i = 10
        return i + 15
    '\n    Launch the GUI.\n    '
    entries = find_failing_tests(result_images, source)
    if len(entries) == 0:
        print('No failed tests')
        sys.exit(0)
    app = QtWidgets.QApplication(sys.argv)
    dialog = Dialog(entries)
    dialog.show()
    filter = EventFilter(dialog)
    app.installEventFilter(filter)
    sys.exit(_exec(app))
if __name__ == '__main__':
    import argparse
    source_dir = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='\nTriage image comparison test failures.\n\nIf no arguments are provided, it assumes you ran the tests at the\ntop-level of a source checkout as `pytest .`.\n\nKeys:\n    left/right: Move between test, expected and diff images\n    up/down:    Move between tests\n    A:          Accept test.  Copy the test result to the source tree.\n    R:          Reject test.  Copy the expected result to the source tree.\n')
    parser.add_argument('result_images', type=Path, nargs='?', default=source_dir / 'result_images', help='The location of the result_images directory')
    parser.add_argument('source', type=Path, nargs='?', default=source_dir, help='The location of the matplotlib source tree')
    args = parser.parse_args()
    launch(args.result_images, args.source)