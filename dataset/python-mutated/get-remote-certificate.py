import re
import os
import sys
import tempfile

def fetch_server_certificate(host, port):
    if False:
        return 10

    def subproc(cmd):
        if False:
            while True:
                i = 10
        from subprocess import Popen, PIPE, STDOUT
        proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        status = proc.wait()
        output = proc.stdout.read()
        return (status, output)

    def strip_to_x509_cert(certfile_contents, outfile=None):
        if False:
            for i in range(10):
                print('nop')
        m = re.search(b'^([-]+BEGIN CERTIFICATE[-]+[\\r]*\\n.*[\\r]*^[-]+END CERTIFICATE[-]+)$', certfile_contents, re.MULTILINE | re.DOTALL)
        if not m:
            return None
        else:
            tn = tempfile.mktemp()
            with open(tn, 'wb') as fp:
                fp.write(m.group(1) + b'\n')
            try:
                tn2 = outfile or tempfile.mktemp()
                (status, output) = subproc('openssl x509 -in "%s" -out "%s"' % (tn, tn2))
                if status != 0:
                    raise RuntimeError('OpenSSL x509 failed with status %s and output: %r' % (status, output))
                with open(tn2, 'rb') as fp:
                    data = fp.read()
                os.unlink(tn2)
                return data
            finally:
                os.unlink(tn)
    if sys.platform.startswith('win'):
        tfile = tempfile.mktemp()
        with open(tfile, 'w') as fp:
            fp.write('quit\n')
        try:
            (status, output) = subproc('openssl s_client -connect "%s:%s" -showcerts < "%s"' % (host, port, tfile))
        finally:
            os.unlink(tfile)
    else:
        (status, output) = subproc('openssl s_client -connect "%s:%s" -showcerts < /dev/null' % (host, port))
    if status != 0:
        raise RuntimeError('OpenSSL connect failed with status %s and output: %r' % (status, output))
    certtext = strip_to_x509_cert(output)
    if not certtext:
        raise ValueError('Invalid response received from server at %s:%s' % (host, port))
    return certtext
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write('Usage:  %s HOSTNAME:PORTNUMBER [, HOSTNAME:PORTNUMBER...]\n' % sys.argv[0])
        sys.exit(1)
    for arg in sys.argv[1:]:
        (host, port) = arg.split(':')
        sys.stdout.buffer.write(fetch_server_certificate(host, int(port)))
    sys.exit(0)