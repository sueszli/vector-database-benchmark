from app import challenges

def test_valid():
    if False:
        while True:
            i = 10
    for challenge in challenges:
        for p in challenge['problems']:
            val_fn = p['validator']
            try:
                val_fn('response', 'input')
            except Exception:
                import traceback
                traceback.print_exc()
                print(p, 'failed')