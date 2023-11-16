"""Checks use of consider-merging-isinstance"""

def isinstances():
    if False:
        i = 10
        return i + 15
    'Examples of isinstances'
    var = range(10)
    if isinstance(var[1], (int, float)):
        pass
    result = isinstance(var[2], (int, float))
    if isinstance(var[3], int) or isinstance(var[3], float) or (isinstance(var[3], list) and True):
        pass
    result = isinstance(var[4], int) or isinstance(var[4], float) or (isinstance(var[5], list) and False)
    result = isinstance(var[5], int) or True or isinstance(var[5], float)
    inferred_isinstance = isinstance
    result = inferred_isinstance(var[6], int) or inferred_isinstance(var[6], float) or (inferred_isinstance(var[6], list) and False)
    result = isinstance(var[10], str) or (isinstance(var[10], int) and var[8] * 14) or (isinstance(var[10], float) and var[5] * 14.4) or isinstance(var[10], list)
    result = isinstance(var[11], int) or isinstance(var[11], int) or isinstance(var[11], float)
    result = isinstance(var[20])
    result = isinstance()
    result = isinstance(var[12], (int, float)) or isinstance(var[12], list)
    result = isinstance(var[5], int) and var[5] * 14 or (isinstance(var[5], float) and var[5] * 14.4)
    result = isinstance(var[7], int) or not isinstance(var[7], float)
    result = isinstance(var[6], int) or isinstance(var[7], float)
    result = isinstance(var[6], int) or isinstance(var[7], int)
    result = isinstance(var[6], (float, int)) or False
    return result
if isinstance(self.k, int) or isinstance(self.k, float):
    ...