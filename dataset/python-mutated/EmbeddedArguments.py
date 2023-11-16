from robot.api.deco import keyword

@keyword(name='${num1:\\d+} + ${num2:\\d+} = ${exp:\\d+}')
def add(num1: int, num2: int, expected: int):
    if False:
        return 10
    result = num1 + num2
    assert result == expected, (result, expected)

@keyword(name='${num1:\\d+} - ${num2:\\d+} = ${exp:\\d+}', types=(int, int, int))
def sub(num1, num2, expected):
    if False:
        while True:
            i = 10
    result = num1 - num2
    assert result == expected, (result, expected)

@keyword(name='${num1:\\d+} * ${num2:\\d+} = ${exp:\\d+}')
def mul(num1=0, num2=0, expected=0):
    if False:
        for i in range(10):
            print('nop')
    result = num1 * num2
    assert result == expected, (result, expected)