import pysnooper

@pysnooper.snoop(depth=2, color=False)
def factorial(x):
    if False:
        i = 10
        return i + 15
    if x <= 1:
        return 1
    return mul(x, factorial(x - 1))

def mul(a, b):
    if False:
        while True:
            i = 10
    return a * b

def main():
    if False:
        i = 10
        return i + 15
    factorial(4)
expected_output = '\nSource path:... Whatever\nStarting var:.. x = 4\n09:31:32.691599 call         5 def factorial(x):\n09:31:32.691722 line         6     if x <= 1:\n09:31:32.691746 line         8     return mul(x, factorial(x - 1))\n    Starting var:.. x = 3\n    09:31:32.691781 call         5 def factorial(x):\n    09:31:32.691806 line         6     if x <= 1:\n    09:31:32.691823 line         8     return mul(x, factorial(x - 1))\n        Starting var:.. x = 2\n        09:31:32.691852 call         5 def factorial(x):\n        09:31:32.691875 line         6     if x <= 1:\n        09:31:32.691892 line         8     return mul(x, factorial(x - 1))\n            Starting var:.. x = 1\n            09:31:32.691918 call         5 def factorial(x):\n            09:31:32.691941 line         6     if x <= 1:\n            09:31:32.691961 line         7         return 1\n            09:31:32.691978 return       7         return 1\n            Return value:.. 1\n            Elapsed time: 00:00:00.000092\n            Starting var:.. a = 2\n            Starting var:.. b = 1\n            09:31:32.692025 call        11 def mul(a, b):\n            09:31:32.692055 line        12     return a * b\n            09:31:32.692075 return      12     return a * b\n            Return value:.. 2\n        09:31:32.692102 return       8     return mul(x, factorial(x - 1))\n        Return value:.. 2\n        Elapsed time: 00:00:00.000283\n        Starting var:.. a = 3\n        Starting var:.. b = 2\n        09:31:32.692147 call        11 def mul(a, b):\n        09:31:32.692174 line        12     return a * b\n        09:31:32.692193 return      12     return a * b\n        Return value:.. 6\n    09:31:32.692216 return       8     return mul(x, factorial(x - 1))\n    Return value:.. 6\n    Elapsed time: 00:00:00.000468\n    Starting var:.. a = 4\n    Starting var:.. b = 6\n    09:31:32.692259 call        11 def mul(a, b):\n    09:31:32.692285 line        12     return a * b\n    09:31:32.692304 return      12     return a * b\n    Return value:.. 24\n09:31:32.692326 return       8     return mul(x, factorial(x - 1))\nReturn value:.. 24\nElapsed time: 00:00:00.000760\n'