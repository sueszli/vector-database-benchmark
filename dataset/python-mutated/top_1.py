"""
This algorithm receives an array and returns most_frequent_value
Also, sometimes it is possible to have multiple 'most_frequent_value's,
so this function returns a list. This result can be used to find a 
representative value in an array.

This algorithm gets an array, makes a dictionary of it,
 finds the most frequent count, and makes the result list.

For example: top_1([1, 1, 2, 2, 3, 4]) will return [1, 2]

(TL:DR) Get mathematical Mode
Complexity: O(n)
"""

def top_1(arr):
    if False:
        print('Hello World!')
    values = {}
    result = []
    f_val = 0
    for i in arr:
        if i in values:
            values[i] += 1
        else:
            values[i] = 1
    f_val = max(values.values())
    for i in values.keys():
        if values[i] == f_val:
            result.append(i)
        else:
            continue
    return result