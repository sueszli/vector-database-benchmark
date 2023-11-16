bharat_expenses = [20, 30, 45]
bilal_expenses = [45, 67, 34]
total = 0
for item in bharat_expenses:
    total += item
print("Bharat's total:", total)
total = 0
for item in bilal_expenses:
    total += item
print("Bilal's total:", total)

def find_total(exp):
    if False:
        return 10
    '\n    This function takes list of numbers as input and returns sum of that list\n    :param exp: input list\n    :return: total sum\n    '
    total = 0
    for item in exp:
        total += item
    return total
bharat_total = find_total(bharat_expenses)
print("Bharat's total:", bharat_total)
bilal_total = find_total(bilal_expenses)
print("Bilal's total:", bilal_total)
print(help(find_total))

def cylinder_volume(radius, height=1):
    if False:
        print('Hello World!')
    print('radius is:', radius)
    print('height is:', height)
    area = 3.14 * radius ** 2 * height
    return area
r = 5
h = 10
print(cylinder_volume(height=h, radius=r))
r = 5
h = 10
print(cylinder_volume(radius=r))