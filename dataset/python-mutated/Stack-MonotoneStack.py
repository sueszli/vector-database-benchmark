import random

def monotoneStack(nums):
    if False:
        print('Hello World!')
    print(str(nums))
    stack = []
    for num in nums:
        while stack and num <= stack[-1]:
            top = stack[-1]
            stack.pop()
            print(str(top) + ' 出栈 ' + str(stack))
        stack.append(num)
        print(str(num) + ' 入栈 ' + str(stack))

def monotoneIncreasingStack(nums):
    if False:
        for i in range(10):
            print('nop')
    stack = []
    for num in nums:
        while stack and num >= stack[-1]:
            top = stack[-1]
            stack.pop()
            print(str(top) + ' 出栈 ' + str(stack))
        stack.append(num)
        print(str(num) + ' 入栈 ' + str(stack))

def monotoneDecreasingStack(nums):
    if False:
        while True:
            i = 10
    stack = []
    for num in nums:
        while stack and num <= stack[-1]:
            top = stack[-1]
            stack.pop()
            print(str(top) + ' 出栈 ' + str(stack))
        stack.append(num)
        print(str(num) + ' 入栈 ' + str(stack))
nums = []
for i in range(8):
    nums.append(random.randint(1, 9))
print(nums)
monotoneIncreasingStack(nums)