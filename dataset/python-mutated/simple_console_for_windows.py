"""Start a simple interactive console with TensorFlow available."""
import code
import sys

def main(_):
    if False:
        for i in range(10):
            print('nop')
    'Run an interactive console.'
    code.interact()
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))