nums = range(1, 101)

def fizzbuzz(nums):
    if False:
        while True:
            i = 10
    for n in nums:
        if (n % 3 and n % 5) == 0:
            print('fizzbuzz')
        elif n % 3 == 0:
            print('fizz')
        elif n % 5 == 0:
            print('buzz')
        else:
            print(n)
fizzbuzz(nums)