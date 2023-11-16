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
"\n------------------------------IMPORTANT NOTE----------------------------------------\n\nThis algorithm gives the result without updating the actual array.\nSo if constraind is given that you don't have to update the array then this is the algorithm you are looking for\n"
'\n------------------------------EXPLANATION----------------------------------------\n\nThe main idea is that we create 2 variable current_max and max_so_far\nwe keep adding all the array element in current_max if at any point current_max\nbecame grater than max_so_far we change the value of max_so_far to current_max value\nAlso if at any point current_so far became negative we reassigned it to 0\n\nFor example \narr = [-2, 3, 2, -2]\ncurrent_max = 0\nmax_so_far = -inf   (-infinite)  \nNow we loop through arr and keep adding arr value to current_max\n\n-----------1st iteration----------\ncurrent_max = -2, maximum_so_far = -inf   (current_max is negative and greater than max_so_far)\nTherefore current_max = 0, maximum_so_far = -2\n\n-----------2nd iteration----------\ncurrent_max = 3 , maximum_so_far = 2   (current_max is positive and its value is > maximum_so_far)\nTherefore maximum_so_far = 3  \n\n-----------3rd iteration----------\ncurrent_max = 5, maximum_so_far = 3\nTherefore maximum_so_far = 5\n\n-----------4th iteration----------\ncurrent_max = 3, maximum_so_far = 5\nSince current_max < maximum_so_far therefore maximum_so_far = 5\n\nAt last we return maximum_so_far    \n'

def max_sub_array(arr):
    if False:
        while True:
            i = 10
    '\n    Time Complexity : O(n)\n    Space Complexity : O(1)\n    '
    current_max = 0
    maximum_so_far = float('-inf')
    for i in arr:
        current_max += i
        if current_max > maximum_so_far:
            maximum_so_far = current_max
        if current_max < 0:
            current_max = 0
    return maximum_so_far
print(max_sub_array([-2, 3, 2, -2]))