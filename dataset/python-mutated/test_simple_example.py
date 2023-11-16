import uiautomator2 as u2

def test_simple():
    if False:
        i = 10
        return i + 15
    d = u2.connect()
    print(d.info)
if __name__ == '__main__':
    test_simple()