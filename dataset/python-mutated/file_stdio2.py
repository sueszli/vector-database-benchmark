import sys
try:
    sys.stdout.buffer
    sys.stdin.buffer
    sys.stderr.buffer
except AttributeError:
    print('SKIP')
    raise SystemExit

def print_flush(*args, **kwargs):
    if False:
        return 10
    try:
        print(*args, **kwargs, flush=True)
    except TypeError:
        print(*args, **kwargs)
print_flush('==stdin==')
print_flush(sys.stdin.buffer.fileno())
print_flush('==stdout==')
print_flush(sys.stdout.buffer.fileno())
n_text = sys.stdout.write('The quick brown fox jumps over the lazy dog\n')
sys.stdout.flush()
n_binary = sys.stdout.buffer.write('The quick brown fox jumps over the lazy dog\n'.encode('utf-8'))
sys.stdout.buffer.flush()
print_flush('n_text:{} n_binary:{}'.format(n_text, n_binary))
print_flush('==stderr==')
print_flush(sys.stderr.buffer.fileno())
n_text = sys.stderr.write('The quick brown fox jumps over the lazy dog\n')
sys.stderr.flush()
n_binary = sys.stderr.buffer.write('The quick brown fox jumps over the lazy dog\n'.encode('utf-8'))
sys.stderr.buffer.flush()
print_flush('n_text:{} n_binary:{}'.format(n_text, n_binary))