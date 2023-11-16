import sys
from remoteserver import RemoteServer, keyword

class KeywordTags:

    def no_tags(self):
        if False:
            i = 10
            return i + 15
        pass

    def doc_contains_tags_only(self):
        if False:
            return 10
        'Tags: foo, bar'

    def doc_contains_tags_after_doc(self):
        if False:
            print('Hello World!')
        'This is by doc.\n\n        My doc has multiple lines.\n\n        Tags: these, are, my, tags\n        '

    @keyword
    def empty_robot_tags_means_no_tags(self):
        if False:
            i = 10
            return i + 15
        pass

    @keyword(tags=['foo', 'bar', 'FOO', '42'])
    def robot_tags(self):
        if False:
            return 10
        pass

    @keyword(tags=['foo', 'bar'])
    def robot_tags_and_doc_tags(self):
        if False:
            print('Hello World!')
        'Tags: bar, zap'
if __name__ == '__main__':
    RemoteServer(KeywordTags(), *sys.argv[1:])