"""
    Icourse163 抽象基类
"""
import os
if __package__ is None:
    import sys
    sys.path.append('../')
from Mooc.Mooc_Base import *
from Mooc.Mooc_Download import *
from Mooc.Mooc_Request import *
from Mooc.Mooc_Potplayer import *
__all__ = ['Icourse163_Base']

class Icourse163_Base(Mooc_Base):
    potplayer = Mooc_Potplayer()

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.infos = {}
        self.__term_id = None

    @property
    def term_id(self):
        if False:
            i = 10
            return i + 15
        return self.__term_id

    @term_id.setter
    def term_id(self, term_id):
        if False:
            print('Hello World!')
        self.__term_id = term_id

    def set_mode(self):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                instr = input('请输入一个0-4的数选择性下载内容(1:超高清, 2:高清, 3:标清, 4:仅下载课件) [0退出]: ')
                if not instr:
                    continue
                try:
                    innum = int(instr)
                    if innum == 0:
                        return False
                    elif 1 <= innum <= 4:
                        self.mode = innum
                        return True
                    else:
                        print('请输入一个0-4之间的整数!')
                        continue
                except ValueError:
                    print('请输入一个0-4之间的整数!')
            except KeyboardInterrupt:
                pass

    @classmethod
    @potplayer
    def download_video(cls, video_url, video_name, video_dir):
        if False:
            while True:
                i = 10
        return super().download_video(video_url, video_name, video_dir)