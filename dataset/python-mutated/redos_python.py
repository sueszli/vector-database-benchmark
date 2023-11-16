import re
re.match('(a+)+$', ...)
re.match('hello+', ...)

def test_patterns(input_str):
    if False:
        for i in range(10):
            print('nop')
    vulnerable = re.match('((honk )+)+$', input_str)
    ok = re.match('((honk )+)++$', input_str)

def validate_email(email_str):
    if False:
        i = 10
        return i + 15
    re.match('^([a-zA-Z0-9])(([\\-.]|[_]+)?([a-zA-Z0-9]+))*(@){1}[a-z0-9]+[.]{1}(([a-z]{2,3})|([a-z]{2,3}[.]{1}[a-z]{2,3}))$', email_str)