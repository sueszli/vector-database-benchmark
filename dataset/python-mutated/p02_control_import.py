"""
Topic: 控制导入内容
Desc : 
"""

def spam():
    if False:
        return 10
    pass

def grok():
    if False:
        while True:
            i = 10
    pass
blah = 42
__all__ = ['spam', 'grok']