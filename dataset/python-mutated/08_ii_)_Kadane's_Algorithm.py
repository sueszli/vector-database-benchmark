"""
EXAMPLE:
    INPUT : [1, 2, 3, -2, 5]
    OUTPUT : 9
    EXPLANATION :
    Because sub-array (1,2,3,-2,5) has maximum sum among all sub-array.
    For example : sub-array (1,2,3) has sum 6
                  sub-array (1,2,3,-2) has sum 4
                  sub-array (3,-2,5) has sum 6
                  and so on..................
                  Final max sum will be 9 and hence we return it
"""
"\n------------------------------IMPORTANT NOTE---------------------------------\n\nThis algorithm update the given array so if you are restricted to update the\ngiven array see _08_i_)_Kadane's_Algorithm\n"
'\n------------------------------EXPLANATION----------------------------------------\n\nThe main idea in this algorithm is: \nEach time we took 2 element (array[n] and array[n + 1]) from the given array and add them\nIf there summation is greater than array[n + 1] then we replace the value of array[n + 1]\nwith the summation.\n\nAfter we finishes with updating the array we simply return the maximum among the\narray\n\nFor example:\narr = [-2, 2, 3, -2]\nNow the loop began:\n\n--------------1st iteration--------------------\nsummation = arr[0] + arr[1] = 0\nis summation > arr[1] \nno so we continue without replacing \n\n--------------2nd iteration--------------------\nsummation = arr[1] + arr[2] = 5\nis summation > arr[2]\nyes so we replace arr[2] with summation\nupdated arr = [-2, 2, 5, -2]\n\n--------------3nd iteration--------------------\nsummation = arr[2] + arr[3] = 3\nis summation > arr[3]\nyes so we replace arr[3] with summation\nupdated arr = [-2, 2, 5, 3]\n\nloop ends\n\nwe return max among the updated array \nwhich is 5\n'

def max_sub_array(arr):
    if False:
        while True:
            i = 10
    '\n    TIME COMPLEXITY : O(n)\n    SPACE COMPLEXITY : O(1)\n    '
    for i in range(len(arr) - 1):
        summation = arr[i] + arr[i + 1]
        arr[i + 1] = max(summation, arr[i + 1])
    return max(arr)
print(max_sub_array([-2, 2, 3, -2]))