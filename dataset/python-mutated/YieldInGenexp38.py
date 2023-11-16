def strangeLambdaGeneratorExpression():
    if False:
        for i in range(10):
            print('nop')
    x = ((yield) for i in (1, 2) if (yield))
    print('Strange lambda generator expression')
    print(list(x))