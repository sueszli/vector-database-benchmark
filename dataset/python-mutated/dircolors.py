import os
import stat
from contextlib import suppress
from typing import Dict, Generator, Optional, Tuple, Union
DEFAULT_DIRCOLORS = '# {{{\n# Configuration file for dircolors, a utility to help you set the\n# LS_COLORS environment variable used by GNU ls with the --color option.\n# Copyright (C) 1996-2019 Free Software Foundation, Inc.\n# Copying and distribution of this file, with or without modification,\n# are permitted provided the copyright notice and this notice are preserved.\n# The keywords COLOR, OPTIONS, and EIGHTBIT (honored by the\n# slackware version of dircolors) are recognized but ignored.\n# Below are TERM entries, which can be a glob patterns, to match\n# against the TERM environment variable to determine if it is colorizable.\nTERM Eterm\nTERM ansi\nTERM *color*\nTERM con[0-9]*x[0-9]*\nTERM cons25\nTERM console\nTERM cygwin\nTERM dtterm\nTERM gnome\nTERM hurd\nTERM jfbterm\nTERM konsole\nTERM kterm\nTERM linux\nTERM linux-c\nTERM mlterm\nTERM putty\nTERM rxvt*\nTERM screen*\nTERM st\nTERM terminator\nTERM tmux*\nTERM vt100\nTERM xterm*\n# Below are the color init strings for the basic file types.\n# One can use codes for 256 or more colors supported by modern terminals.\n# The default color codes use the capabilities of an 8 color terminal\n# with some additional attributes as per the following codes:\n# Attribute codes:\n# 00=none 01=bold 04=underscore 05=blink 07=reverse 08=concealed\n# Text color codes:\n# 30=black 31=red 32=green 33=yellow 34=blue 35=magenta 36=cyan 37=white\n# Background color codes:\n# 40=black 41=red 42=green 43=yellow 44=blue 45=magenta 46=cyan 47=white\n#NORMAL 00 # no color code at all\n#FILE 00 # regular file: use no color at all\nRESET 0 # reset to "normal" color\nDIR 01;34 # directory\nLINK 01;36 # symbolic link. (If you set this to \'target\' instead of a\n # numerical value, the color is as for the file pointed to.)\nMULTIHARDLINK 00 # regular file with more than one link\nFIFO 40;33 # pipe\nSOCK 01;35 # socket\nDOOR 01;35 # door\nBLK 40;33;01 # block device driver\nCHR 40;33;01 # character device driver\nORPHAN 40;31;01 # symlink to nonexistent file, or non-stat\'able file ...\nMISSING 00 # ... and the files they point to\nSETUID 37;41 # file that is setuid (u+s)\nSETGID 30;43 # file that is setgid (g+s)\nCAPABILITY 30;41 # file with capability\nSTICKY_OTHER_WRITABLE 30;42 # dir that is sticky and other-writable (+t,o+w)\nOTHER_WRITABLE 34;42 # dir that is other-writable (o+w) and not sticky\nSTICKY 37;44 # dir with the sticky bit set (+t) and not other-writable\n# This is for files with execute permission:\nEXEC 01;32\n# List any file extensions like \'.gz\' or \'.tar\' that you would like ls\n# to colorize below. Put the extension, a space, and the color init string.\n# (and any comments you want to add after a \'#\')\n# If you use DOS-style suffixes, you may want to uncomment the following:\n#.cmd 01;32 # executables (bright green)\n#.exe 01;32\n#.com 01;32\n#.btm 01;32\n#.bat 01;32\n# Or if you want to colorize scripts even if they do not have the\n# executable bit actually set.\n#.sh 01;32\n#.csh 01;32\n # archives or compressed (bright red)\n.tar 01;31\n.tgz 01;31\n.arc 01;31\n.arj 01;31\n.taz 01;31\n.lha 01;31\n.lz4 01;31\n.lzh 01;31\n.lzma 01;31\n.tlz 01;31\n.txz 01;31\n.tzo 01;31\n.t7z 01;31\n.zip 01;31\n.z 01;31\n.dz 01;31\n.gz 01;31\n.lrz 01;31\n.lz 01;31\n.lzo 01;31\n.xz 01;31\n.zst 01;31\n.tzst 01;31\n.bz2 01;31\n.bz 01;31\n.tbz 01;31\n.tbz2 01;31\n.tz 01;31\n.deb 01;31\n.rpm 01;31\n.jar 01;31\n.war 01;31\n.ear 01;31\n.sar 01;31\n.rar 01;31\n.alz 01;31\n.ace 01;31\n.zoo 01;31\n.cpio 01;31\n.7z 01;31\n.rz 01;31\n.cab 01;31\n.wim 01;31\n.swm 01;31\n.dwm 01;31\n.esd 01;31\n# image formats\n.jpg 01;35\n.jpeg 01;35\n.mjpg 01;35\n.mjpeg 01;35\n.gif 01;35\n.bmp 01;35\n.pbm 01;35\n.pgm 01;35\n.ppm 01;35\n.tga 01;35\n.xbm 01;35\n.xpm 01;35\n.tif 01;35\n.tiff 01;35\n.png 01;35\n.svg 01;35\n.svgz 01;35\n.mng 01;35\n.pcx 01;35\n.mov 01;35\n.mpg 01;35\n.mpeg 01;35\n.m2v 01;35\n.mkv 01;35\n.webm 01;35\n.ogm 01;35\n.mp4 01;35\n.m4v 01;35\n.mp4v 01;35\n.vob 01;35\n.qt 01;35\n.nuv 01;35\n.wmv 01;35\n.asf 01;35\n.rm 01;35\n.rmvb 01;35\n.flc 01;35\n.avi 01;35\n.fli 01;35\n.flv 01;35\n.gl 01;35\n.dl 01;35\n.xcf 01;35\n.xwd 01;35\n.yuv 01;35\n.cgm 01;35\n.emf 01;35\n# https://wiki.xiph.org/MIME_Types_and_File_Extensions\n.ogv 01;35\n.ogx 01;35\n# audio formats\n.aac 00;36\n.au 00;36\n.flac 00;36\n.m4a 00;36\n.mid 00;36\n.midi 00;36\n.mka 00;36\n.mp3 00;36\n.mpc 00;36\n.ogg 00;36\n.ra 00;36\n.wav 00;36\n# https://wiki.xiph.org/MIME_Types_and_File_Extensions\n.oga 00;36\n.opus 00;36\n.spx 00;36\n.xspf 00;36\n'
special_types = ((stat.S_IFLNK, 'ln'), (stat.S_IFIFO, 'pi'), (stat.S_IFSOCK, 'so'), (stat.S_IFBLK, 'bd'), (stat.S_IFCHR, 'cd'), (stat.S_ISUID, 'su'), (stat.S_ISGID, 'sg'))
CODE_MAP = {'RESET': 'rs', 'DIR': 'di', 'LINK': 'ln', 'MULTIHARDLINK': 'mh', 'FIFO': 'pi', 'SOCK': 'so', 'DOOR': 'do', 'BLK': 'bd', 'CHR': 'cd', 'ORPHAN': 'or', 'MISSING': 'mi', 'SETUID': 'su', 'SETGID': 'sg', 'CAPABILITY': 'ca', 'STICKY_OTHER_WRITABLE': 'tw', 'OTHER_WRITABLE': 'ow', 'STICKY': 'st', 'EXEC': 'ex'}

def stat_at(file: str, cwd: Optional[Union[int, str]]=None, follow_symlinks: bool=False) -> os.stat_result:
    if False:
        for i in range(10):
            print('nop')
    dirfd: Optional[int] = None
    need_to_close = False
    if isinstance(cwd, str):
        dirfd = os.open(cwd, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
        need_to_close = True
    elif isinstance(cwd, int):
        dirfd = cwd
    try:
        return os.stat(file, dir_fd=dirfd, follow_symlinks=follow_symlinks)
    finally:
        if need_to_close and dirfd is not None:
            os.close(dirfd)

class Dircolors:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.codes: Dict[str, str] = {}
        self.extensions: Dict[str, str] = {}
        if not self.load_from_environ() and (not self.load_from_file()):
            self.load_defaults()

    def clear(self) -> None:
        if False:
            print('Hello World!')
        self.codes.clear()
        self.extensions.clear()

    def load_from_file(self) -> bool:
        if False:
            print('Hello World!')
        for candidate in (os.path.expanduser('~/.dir_colors'), '/etc/DIR_COLORS'):
            with suppress(Exception):
                with open(candidate) as f:
                    return self.load_from_dircolors(f.read())
        return False

    def load_from_lscolors(self, lscolors: str) -> bool:
        if False:
            i = 10
            return i + 15
        self.clear()
        if not lscolors:
            return False
        for item in lscolors.split(':'):
            try:
                (code, color) = item.split('=', 1)
            except ValueError:
                continue
            if code.startswith('*.'):
                self.extensions[code[1:]] = color
            else:
                self.codes[code] = color
        return bool(self.codes or self.extensions)

    def load_from_environ(self, envvar: str='LS_COLORS') -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.load_from_lscolors(os.environ.get(envvar) or '')

    def load_from_dircolors(self, database: str, strict: bool=False) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self.clear()
        for line in database.splitlines():
            line = line.split('#')[0].strip()
            if not line:
                continue
            split = line.split()
            if len(split) != 2:
                if strict:
                    raise ValueError(f'Warning: unable to parse dircolors line "{line}"')
                continue
            (key, val) = split
            if key == 'TERM':
                continue
            if key in CODE_MAP:
                self.codes[CODE_MAP[key]] = val
            elif key.startswith('.'):
                self.extensions[key] = val
            elif strict:
                raise ValueError(f'Warning: unable to parse dircolors line "{line}"')
        return bool(self.codes or self.extensions)

    def load_defaults(self) -> bool:
        if False:
            return 10
        self.clear()
        return self.load_from_dircolors(DEFAULT_DIRCOLORS, True)

    def generate_lscolors(self) -> str:
        if False:
            i = 10
            return i + 15
        ' Output the database in the format used by the LS_COLORS environment variable. '

        def gen_pairs() -> Generator[Tuple[str, str], None, None]:
            if False:
                print('Hello World!')
            for pair in self.codes.items():
                yield pair
            for pair in self.extensions.items():
                yield ('*' + pair[0], pair[1])
        return ':'.join(('{}={}'.format(*pair) for pair in gen_pairs()))

    def _format_code(self, text: str, code: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        val = self.codes.get(code)
        return '\x1b[{}m{}\x1b[{}m'.format(val, text, self.codes.get('rs', '0')) if val else text

    def _format_ext(self, text: str, ext: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        val = self.extensions.get(ext, '0')
        return '\x1b[{}m{}\x1b[{}m'.format(val, text, self.codes.get('rs', '0')) if val else text

    def format_mode(self, text: str, sr: os.stat_result) -> str:
        if False:
            print('Hello World!')
        mode = sr.st_mode
        if stat.S_ISDIR(mode):
            if mode & (stat.S_ISVTX | stat.S_IWOTH) == stat.S_ISVTX | stat.S_IWOTH:
                return self._format_code(text, 'tw')
            if mode & stat.S_ISVTX:
                return self._format_code(text, 'st')
            if mode & stat.S_IWOTH:
                return self._format_code(text, 'ow')
            return self._format_code(text, 'di')
        for (mask, code) in special_types:
            if mode & mask == mask:
                return self._format_code(text, code)
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return self._format_code(text, 'ex')
        ext = os.path.splitext(text)[1]
        if ext:
            return self._format_ext(text, ext)
        return text

    def __call__(self, path: str, text: str, cwd: Optional[Union[int, str]]=None) -> str:
        if False:
            return 10
        follow_symlinks = self.codes.get('ln') == 'target'
        try:
            sr = stat_at(path, cwd, follow_symlinks)
        except OSError:
            return text
        return self.format_mode(text, sr)

def develop() -> None:
    if False:
        i = 10
        return i + 15
    import sys
    print(Dircolors()(sys.argv[-1], sys.argv[-1]))