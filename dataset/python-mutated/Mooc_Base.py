"""
    Mooc 的虚基类：用于派生所有Mooc子类
"""
import os
from abc import ABC, abstractmethod
from Mooc.Mooc_Config import *
from Mooc.Mooc_Download import *
from Mooc.Mooc_Request import *
__all__ = ['Mooc_Base']

class Mooc_Base(ABC):

    def __init__(self):
        if False:
            return 10
        self.__mode = None
        self.__cid = None
        self.__title = None
        self.__infos = None
        self.__rootDir = None

    @property
    def mode(self):
        if False:
            for i in range(10):
                print('nop')
        '下载模式: 用于选择性下载'
        return self.__mode

    @mode.setter
    def mode(self, mode):
        if False:
            return 10
        self.__mode = mode

    @property
    def cid(self):
        if False:
            for i in range(10):
                print('nop')
        '课程的 ID'
        return self.__cid

    @cid.setter
    def cid(self, cid):
        if False:
            while True:
                i = 10
        self.__cid = cid

    @property
    def title(self):
        if False:
            print('Hello World!')
        '课程的标题'
        return self.__title

    @title.setter
    def title(self, title):
        if False:
            while True:
                i = 10
        self.__title = title

    @property
    def infos(self):
        if False:
            for i in range(10):
                print('nop')
        '解析后的课程信息'
        return self.__infos

    @property
    def rootDir(self):
        if False:
            return 10
        return self.__rootDir

    @rootDir.setter
    def rootDir(self, rootDir):
        if False:
            i = 10
            return i + 15
        self.__rootDir = rootDir

    @infos.setter
    def infos(self, infos):
        if False:
            while True:
                i = 10
        self.__infos = infos

    @abstractmethod
    def _get_cid(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def _get_title(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _get_infos(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _download(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def set_mode(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def prepare(self, url):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def download(self):
        if False:
            print('Hello World!')
        pass

    @classmethod
    def download_video(cls, video_url, video_name, video_dir):
        if False:
            while True:
                i = 10
        '下载 MP4 视频文件'
        succeed = True
        if not cls.judge_file_existed(video_dir, video_name, '.mp4'):
            try:
                header = request_head(video_url)
                size = float(header['Content-Length']) / (1024 * 1024)
                print('  |-{}  [mp4] 大小: {:.2f}M'.format(cls.align(video_name, LENGTH), size))
                aria2_download_file(video_url, video_name + '.mp4', video_dir)
            except DownloadFailed:
                print('  |-{}  [mp4] 资源无法下载！'.format(cls.align(video_name, LENGTH)))
                succeed = False
        else:
            print('  |-{}  [mp4] 已经成功下载！'.format(cls.align(video_name, LENGTH)))
        return succeed

    @classmethod
    def download_pdf(cls, pdf_url, pdf_name, pdf_dir):
        if False:
            print('Hello World!')
        '下载 PDF '
        succeed = True
        if not cls.judge_file_existed(pdf_dir, pdf_name, '.pdf'):
            try:
                aria2_download_file(pdf_url, pdf_name + '.pdf', pdf_dir)
                print('  |-{}  (pdf) 已经成功下载！'.format(cls.align(pdf_name, LENGTH)))
            except DownloadFailed:
                print('  |-{}  (pdf) 资源无法下载！'.format(cls.align(pdf_name, LENGTH)))
                succeed = False
        else:
            print('  |-{}  (pdf) 已经成功下载！'.format(cls.align(pdf_name, LENGTH)))
        return succeed

    @classmethod
    def download_sub(cls, sub_url, sub_name, sub_dir):
        if False:
            print('Hello World!')
        '下载字幕'
        succeed = True
        if not cls.judge_file_existed(sub_dir, sub_name, '.srt'):
            try:
                aria2_download_file(sub_url, sub_name + '.srt', sub_dir)
            except DownloadFailed:
                succeed = False
        return succeed

    @staticmethod
    def judge_file_existed(dirname, filename, fmt):
        if False:
            return 10
        '\n        judge_file_existed(dirname, filename, fmt) \n        判断在 dirname 目录下是否存在已下载成功的格式为 fmt 且文件名为 filename 的文件\n        '
        filepath = os.path.join(dirname, filename)
        exist1 = os.path.exists(filepath + fmt)
        exist2 = os.path.exists(filepath + fmt + '.aria2')
        return exist1 and (not exist2)

    @staticmethod
    def align(string, width):
        if False:
            for i in range(10):
                print('nop')
        '\n        align(string, width) 根据width宽度居中对齐字符窜 string，主要用于汉字居中\n        '
        res = ''
        size = 0
        for ch in string:
            if size + 3 > width:
                break
            size += 1 if ord(ch) <= 127 else 2
            res += ch
        res += (width - size) * ' '
        return res