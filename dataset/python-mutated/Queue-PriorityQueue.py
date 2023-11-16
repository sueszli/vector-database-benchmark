class Heapq:

    def heapAdjust(self, nums: [int], index: int, end: int):
        if False:
            return 10
        left = index * 2 + 1
        right = left + 1
        while left <= end:
            max_index = index
            if nums[left] > nums[max_index]:
                max_index = left
            if right <= end and nums[right] > nums[max_index]:
                max_index = right
            if index == max_index:
                break
            (nums[index], nums[max_index]) = (nums[max_index], nums[index])
            index = max_index
            left = index * 2 + 1
            right = left + 1

    def heapify(self, nums: [int]):
        if False:
            while True:
                i = 10
        size = len(nums)
        for i in range((size - 2) // 2, -1, -1):
            self.heapAdjust(nums, i, size - 1)

    def heappush(self, nums: list, value):
        if False:
            for i in range(10):
                print('nop')
        nums.append(value)
        size = len(nums)
        i = size - 1
        while (i - 1) // 2 >= 0:
            cur_root = (i - 1) // 2
            if nums[cur_root] > value:
                break
            nums[i] = nums[cur_root]
            i = cur_root
        nums[i] = value

    def heappop(self, nums: list) -> int:
        if False:
            return 10
        size = len(nums)
        (nums[0], nums[-1]) = (nums[-1], nums[0])
        top = nums.pop()
        if size > 0:
            self.heapAdjust(nums, 0, size - 2)
        return top

    def heapSort(self, nums: [int]):
        if False:
            return 10
        self.heapify(nums)
        size = len(nums)
        for i in range(size):
            (nums[0], nums[size - i - 1]) = (nums[size - i - 1], nums[0])
            self.heapAdjust(nums, 0, size - i - 2)
        return nums
nums = [49, 38, 65, 97, 76, 13, 27, 49]
heap = Heapq()
heap.heapSort(nums)
heap.heapify(nums)
rst = heap.heappop(nums)
print(rst)
rst = heap.heappop(nums)
print(rst)
rst = heap.heappop(nums)
print(rst)
rst = heap.heappop(nums)
print(rst)
rst = heap.heappop(nums)
print(rst)
rst = heap.heappop(nums)
print(rst)
nums = [49, 38, 65, 97, 76, 13, 27, 49]
heapList = []
for num in nums:
    heap.heappush(heapList, num)
    print(heapList)
rst = heap.heapSort(heapList)
print(heapList)