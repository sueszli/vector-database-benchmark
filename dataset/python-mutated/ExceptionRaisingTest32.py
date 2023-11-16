def raisy():
    if False:
        i = 10
        return i + 15
    raise ValueError() from None
try:
    print("Raising exception in a function 'from None':")
    raisy()
except (ValueError, TypeError) as e:
    print('Caught as', repr(e))