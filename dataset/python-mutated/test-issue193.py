import time

def test_cls_in_locals():
    if False:
        return 10
    cls = 'This value is not a class'
    time.sleep(0.5)
if __name__ == '__main__':
    test_cls_in_locals()