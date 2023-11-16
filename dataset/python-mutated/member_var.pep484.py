class C:

    def __init__(self, x: Union[complex, float, int, long]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.y = 1 + x