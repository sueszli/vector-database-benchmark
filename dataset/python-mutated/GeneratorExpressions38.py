""" Test case with generator expression form only allow until Python 3.7

"""

def strangeLambdaGeneratorExpression():
    if False:
        i = 10
        return i + 15
    x = ((yield) for i in (1, 2) if (yield))
    print('Strange lambda generator expression:')
    print(list(x))
strangeLambdaGeneratorExpression()