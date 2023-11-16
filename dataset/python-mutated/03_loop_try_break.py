while True:
    try:
        x = 1
        break
    except Exception:
        pass
while True:
    try:
        x -= 1
    except Exception:
        break
for i in range(5):
    try:
        x = 1
        break
    except Exception:
        if i == 4:
            raise

def connect_ws_with_retry(f1, f2):
    if False:
        i = 10
        return i + 15
    while True:
        try:
            f1()
        except Exception:
            f2()
            continue