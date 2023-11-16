from b2.build import type

def register():
    if False:
        return 10
    type.register_type('PREPROCESSED_C', ['i'], 'C')
    type.register_type('PREPROCESSED_CPP', ['ii'], 'CPP')
register()