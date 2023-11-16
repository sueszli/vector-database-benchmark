from matplotlib import pyplot as plt

def range4():
    if False:
        for i in range(10):
            print('nop')
    'Never called if plot_directive works as expected.'
    raise NotImplementedError

def range6():
    if False:
        print('Hello World!')
    'The function that should be executed.'
    plt.figure()
    plt.plot(range(6))
    plt.show()

def range10():
    if False:
        print('Hello World!')
    'The function that should be executed.'
    plt.figure()
    plt.plot(range(10))
    plt.show()