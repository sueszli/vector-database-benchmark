"""
Question link : https://leetcode.com/problems/merge-intervals/
"""
'\n-----------------------------EXPLANATION-----------------------------------\nThe main problem in solving this question is that intervals are randomly distributed in an array\nit means that if we took one intervals then there are lot of possibility :- may be it is merging with \nthe last interval, maybe mid interval and maybe it is not merging at all\n\nFor example : Let arr = [[1, 3], [20, 30], [2, 6], [15, 35], [4, 6]]\n\nThe first interval [1, 3] is merging with 3rd [2, 6] and last interval [4, 6]\nNow how we (a human) know that the first interval is merging with the 3rd and last interval. The answer\nis simple we scan the whole array and find all the intervals which are merging with the first interval.\nNow computer can also use this approach to find the solution. But the time complexity of that approach will\nbe O(n^2) because we are scanning the whole array for a each interval. But we want more efficient solution\n\nSo how can we solve this:\nIf some how array are arranged in such a way that each merging intervals are next to each other then we\nknow where to look for a particular interval\n\nFor example : [[1, 3], [2, 6], [4, 6], [15, 35], [20, 30]]   (we sort the array)\n\nNow if we took interval [1, 3] and compare it with [2, 6] we can easily say that it is merging.\nAfter merging it the interval became [1, 6] now we compare it with [4, 6] and again it merging and so on\n\nSo how just by arranging an array this question became less difficult to solve.\n\n'
"\n-------------------------------------------APPROACH---------------------\n-We sort the given array\n-We create new array for storing updated intervals. Initial value of this array will be the first element of arr\n-Now we began the loop from arr[1:] because we already stored the 1st interval in updated array and since\nthe array is sorted we know it's the right interval\n-Now we compare the current interval from the arr to the store interval in updated interval to see if they are merging\nor not.\n-Now there are 2 type of merging. Either we merge the interval to form the new interval or drop the interval\nentirely\nFor example : [1, 4] and [2, 6]. Here 2 < 3 then there will definitely be a merging and since 4 < 6 therefore\nwe update the arr with interval [1, 6]. But if we are given [1, 4] and [2, 3] since 2 < 4 but 4 > 3 therefore\nwe have to drop the [2, 3] and the array will remain the same\n-That's what we need to check in the if statement\n-But if there is no merging we simply add that interval to the updated interval\n-At last we return the updated_interval\n"

def merge(arr):
    if False:
        return 10
    '\n    Time Complexity : O(nlogn)\n    Auxiliary Space Complexity : O(n)\n    '
    arr.sort()
    updated_interval = [arr[0]]
    current_index = 0
    for intervals in arr[1:]:
        if intervals[0] <= updated_interval[current_index][1] < intervals[1]:
            updated_interval[current_index] = [updated_interval[current_index][0], intervals[1]]
        elif updated_interval[current_index][1] < intervals[0]:
            updated_interval.append(intervals)
            current_index += 1
    return updated_interval
print(merge([[1, 4], [4, 5], [6, 10], [5, 11]]))