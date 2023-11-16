"""
读取CSS用模块。
"""

class CommonHelper:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def readQss(style):
        if False:
            i = 10
            return i + 15
        with open(style, 'r') as f:
            return f.read()