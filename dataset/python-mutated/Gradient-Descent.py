"""
Assume that the error function is x^5 - 2x^3 - 2
To know the slope of any given x value, we take its derivative: 5x^4 - 6x^2
"""
current_x = 0.5
learning_rate = 0.01
epoch = 50

def slope(x):
    if False:
        while True:
            i = 10
    return 5 * x ** 4 - 6 * x ** 2
for i in range(epoch):
    previous_x = current_x
    current_x += -learning_rate * slope(previous_x)
    print(previous_x)
print('The local minimum occurs at %f' % current_x)
'\nThe trick is the learning_rate. By going in the opposite direction of the slope, it approaches the minimum. Additionally, the closer it gets to the minimum, the smaller the slope gets. This reduces each step as the slope approaches 0.\n'