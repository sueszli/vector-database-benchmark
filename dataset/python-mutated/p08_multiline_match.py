"""
Topic: 多行匹配
Desc : 
"""
import re

def multiline_match():
    if False:
        i = 10
        return i + 15
    comment = re.compile('/\\*(.*?)\\*/')
    text1 = '/* this is a comment */'
    text2 = '/* this is a\n    multiline comment */\n    '
    print(comment.findall(text1))
    print(comment.findall(text2))
    comment = re.compile('/\\*((?:.|\\n)*?)\\*/')
    print(comment.findall(text2))
    comment = re.compile('/\\*(.*?)\\*/', re.DOTALL)
    print(comment.findall(text2))
if __name__ == '__main__':
    multiline_match()