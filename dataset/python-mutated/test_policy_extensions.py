from oso import Oso

def test_parses():
    if False:
        return 10
    oso = Oso()
    oso.register_class(type('User', (), {}))
    oso.register_class(type('Repository', (), {}))
    oso.load_files(['write-rules.polar'])