class CMD:
    pass

def generate_stub() -> None:
    if False:
        while True:
            i = 10
    from kittens.tui.operations import as_type_stub
    from kitty.conf.utils import save_type_stub
    text = as_type_stub()
    save_type_stub(text, __file__)