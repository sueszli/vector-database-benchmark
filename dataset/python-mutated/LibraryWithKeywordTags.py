from robot.api.deco import keyword

def library_keyword_tags_with_attribute():
    if False:
        i = 10
        return i + 15
    pass
library_keyword_tags_with_attribute.robot_tags = ['first', 'second']

@keyword(tags=('one', 2, '2', ''))
def library_keyword_tags_with_decorator():
    if False:
        i = 10
        return i + 15
    pass

def library_keyword_tags_with_documentation():
    if False:
        for i in range(10):
            print('nop')
    'Summary line\n\n    Tags: are read only from the last line\n\n    Tags: one, two words'
    pass

@keyword(tags=['one', 2])
def library_keyword_tags_with_documentation_and_attribute():
    if False:
        for i in range(10):
            print('nop')
    'Tags: one, two words'
    pass

@keyword(tags=42)
def invalid_library_keyword_tags():
    if False:
        for i in range(10):
            print('nop')
    pass