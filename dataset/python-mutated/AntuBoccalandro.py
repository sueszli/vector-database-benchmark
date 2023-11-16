def solution_one():
    if False:
        while True:
            i = 10
    for num in range(1, 101):
        if num % 3 == 0 and num % 5 == 0:
            print('fizzbuzz')
        elif num % 3 == 0:
            print('fizz')
        elif num % 5 == 0:
            print('buzz')
        else:
            print(num)

def solution_two():
    if False:
        for i in range(10):
            print('nop')
    for num in range(1, 101):
        print('fizzbuz') if num % 3 == 0 and num % 5 == 0 else print('fizz') if num % 3 == 0 else print('buzz') if num % 5 == 0 else print(num)
if __name__ == '__main__':
    solution_one()
    solution_two()