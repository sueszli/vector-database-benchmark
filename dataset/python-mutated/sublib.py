def keyword_from_deeper_submodule():
    if False:
        i = 10
        return i + 15
    return 'hi again'

class Sub:

    def keyword_from_class_in_deeper_submodule(self):
        if False:
            i = 10
            return i + 15
        return 'bye'