import pysnooper

@pysnooper.snoop(depth=2, color=False)
def main():
    if False:
        return 10
    f2()

def f2():
    if False:
        for i in range(10):
            print('nop')
    f3()

def f3():
    if False:
        for i in range(10):
            print('nop')
    f4()

@pysnooper.snoop(depth=2, color=False)
def f4():
    if False:
        i = 10
        return i + 15
    f5()

def f5():
    if False:
        for i in range(10):
            print('nop')
    pass
expected_output = '\nSource path:... Whatever\n21:10:42.298924 call         5 def main():\n21:10:42.299158 line         6     f2()\n    21:10:42.299205 call         9 def f2():\n    21:10:42.299246 line        10     f3()\n        Source path:... Whatever\n        21:10:42.299305 call        18 def f4():\n        21:10:42.299348 line        19     f5()\n            21:10:42.299386 call        22 def f5():\n            21:10:42.299424 line        23     pass\n            21:10:42.299460 return      23     pass\n            Return value:.. None\n        21:10:42.299509 return      19     f5()\n        Return value:.. None\n        Elapsed time: 00:00:00.000134\n    21:10:42.299577 return      10     f3()\n    Return value:.. None\n21:10:42.299627 return       6     f2()\nReturn value:.. None\nElapsed time: 00:00:00.000885\n'