class StaticAndClassMethod:

    @staticmethod
    def static_method(arg: int):
        if False:
            print('Hello World!')
        assert arg == 42

    @classmethod
    def class_method(cls, arg: int):
        if False:
            for i in range(10):
                print('nop')
        assert arg == 42