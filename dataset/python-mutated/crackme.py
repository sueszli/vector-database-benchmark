import random
import string
chars = string.ascii_uppercase + string.digits
antitrace = False
password = 'SCRT'
PROGRAM = "\n/* This program parses a command line argument.\n *\n * Compile with :\n *   $ gcc -static -O0 crackme.c -o crackme\n *\n * Analyze it with:\n *   $ manticore crackme\n *\n *   - By default, Manticore will consider all input of stdin to be symbolic\n *     It will explore all possible paths, eventually finding the SCRT key\n *\n * Expected output:\n *  $ manticore --proc 5 crackme\n *  2017-04-22 10:57:07,913: [11918] MAIN:INFO: Loading program: ['crackme']\n *  2017-04-22 10:57:07,918: [11918] MAIN:INFO: Workspace: ./mcore_fZKdZ8\n *  2017-04-22 10:57:56,068: [11969][23] EXECUTOR:INFO: Generating testcase No. 1 for state No.23 - Program finished correctly\n *  2017-04-22 10:57:56,461: [11975][21] EXECUTOR:INFO: Generating testcase No. 2 for state No.21 - Program finished correctly\n *  2017-04-22 10:57:56,877: [11978][31] EXECUTOR:INFO: Generating testcase No. 3 for state No.31 - Program finished correctly\n *  2017-04-22 10:57:57,053: [11971][35] EXECUTOR:INFO: Generating testcase No. 4 for state No.35 - Program finished correctly\n *  2017-04-22 10:57:57,817: [11970][42] EXECUTOR:INFO: Generating testcase No. 5 for state No.42 - Program finished correctly\n *  2017-04-22 10:58:26,874: [11975][30] EXECUTOR:INFO: Generating testcase No. 6 for state No.30 - Program finished correctly\n *  2017-04-22 10:58:27,187: [11969][44] EXECUTOR:INFO: Generating testcase No. 7 for state No.44 - Program finished correctly\n *  2017-04-22 10:58:27,571: [11971][27] EXECUTOR:INFO: Generating testcase No. 8 for state No.27 - Program finished correctly\n *  2017-04-22 10:58:28,567: [11978][53] EXECUTOR:INFO: Generating testcase No. 9 for state No.53 - Program finished correctly\n *  2017-04-22 10:58:33,148: [11970][51] EXECUTOR:INFO: Generating testcase No. 10 for state No.51 - Program finished correctly\n *\n *  Look at ./mcore_IJ2sPb for results, you will find something like this:\n *\n *  $ head -c 4 *.stdin\n *  ==> test_00000000.stdin <==\n *  �CMM\n *  ==> test_00000001.stdin <==\n *  �C��\n *  ==> test_00000002.stdin <==\n *  ��SS\n *  ==> test_00000003.stdin <==\n *  ����\n *  ==> test_00000004.stdin <==\n *  SCR\n *  ==> test_00000005.stdin <==\n *  S�TT\n *  ==> test_00000006.stdin <==\n *  SCRT\n *  ==> test_00000007.stdin <==\n *  S���\n *  ==> test_00000008.stdin <==\n *  SC�@\n *  ==> test_00000009.stdin <==\n *  SC�8\n *\n*/\n#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n#include <sys/types.h>\n"
if antitrace:
    PROGRAM += '\n\n#include <sys/ptrace.h>\n#include <sys/wait.h>\nchar brand[] = "http://www.julioauto.com/rants/anti_ptrace.htm";\nvoid anti_ptrace(void)\n{\n    pid_t child;\n\n    if(getenv("LD_PRELOAD"))\n        while(1);\n\n\n    child = fork();\n    if (child){\n        wait(NULL);\n    }else {\n       if (ptrace(PTRACE_TRACEME, 0, 1, 0) == -1)\n            while(1);\n       exit(0);\n    }\n\n   if (ptrace(PTRACE_TRACEME, 0, 0, 0) == -1)\n        while(1);\n\n}\n'
PROGRAM += '\nint\nmain(int argc, char* argv[]){'
if antitrace:
    PROGRAM += '\n    sleep(10);\n    anti_ptrace();\n'
pad = ''.join((random.choice(chars) for _ in range(len(password))))
banner = 'Please enter your password:\n'
import json
PROGRAM += 'printf ("%s");' % json.dumps(banner).strip('"')
PROGRAM += 'char xor(char a, char b){\n    return a^b;\n}\n'
PROGRAM += 'int c;\n'

def func(password, pad, flag=True):
    if False:
        print('Hello World!')
    if len(password) == 1:
        if flag:
            SUBPROGRAMTRUE = '    printf("You are in!\\n");\n'
        else:
            SUBPROGRAMTRUE = '    printf("You are NOT in!\\n");\n'
    else:
        SUBPROGRAMTRUE = func(password[1:], pad[1:], flag)
    if len(password) == 1:
        SUBPROGRAMFALSE = '    printf("You are NOT in!\\n");\n'
    else:
        SUBPROGRAMFALSE = func(''.join((random.choice(chars) for _ in range(len(password) // 2))), pad[1:], False)
    config = random.choice([(True, SUBPROGRAMTRUE, SUBPROGRAMFALSE), (False, SUBPROGRAMFALSE, SUBPROGRAMTRUE)])
    SUBPROGRAM = ''
    if config[0]:
        SUBPROGRAM += "if ( ((c = getchar(), (c >= 0)) && xor(c, '%c') == ('%c' ^ '%c')) ){\n" % (pad[0], password[0], pad[0])
    else:
        SUBPROGRAM += "if ( ((c = getchar(), (c <  0)) || xor(c, '%c') != ('%c' ^ '%c')) ){\n" % (pad[0], password[0], pad[0])
    SUBPROGRAM += config[1]
    SUBPROGRAM += '}else {\n'
    SUBPROGRAM += config[2]
    SUBPROGRAM += '}'
    SUBPROGRAM = ('\n' + '    ').join(SUBPROGRAM.split('\n'))
    return '    ' + SUBPROGRAM + '\n'
PROGRAM += func(password, pad)
PROGRAM += 'return 0;\n}'
print(PROGRAM)