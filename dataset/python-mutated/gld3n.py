""" 
Not an accurate use of the characteristics of each house.
I apologize beforehand for any inconvenience.
This is just an implementation of the excercise.
Also, I'd love to hear or read any suggestions in order to improve the code.  
"""
from collections import Counter
from time import sleep
from random import choice
houses = {'Hufflepuff': 0, 'Slytherin': 0, 'Gryffindor': 0, 'Ravenclaw': 0}
'\nFunction that evaluates any type of input the user \ncould give in order to avoid any error. Then, based on \nthe response, adds a point to the corresponding house.\n'

def verification():
    if False:
        for i in range(10):
            print('nop')
    global houses
    try:
        election = int(input(''))
    except ValueError:
        print('You must provide a *number* between 1 and 4')
        return verification()
    else:
        if election < 1 or election > 4:
            print('You must provide a number between 1 and 4')
            return verification()
    if election == 1:
        houses['Hufflepuff'] += 1
    elif election == 2:
        houses['Slytherin'] += 1
    elif election == 3:
        houses['Gryffindor'] += 1
    else:
        houses['Ravenclaw'] += 1
    print('================================================================================================')

def question_one():
    if False:
        return 10
    print('\n*First question*\nYou consider yourself:\n    1. Optimistic\n    2. Cunning\n    3. extrovert\n    4. Imaginative\n    (Please choose by number.)')
    verification()

def question_two():
    if False:
        i = 10
        return i + 15
    print('\n*Second question*\nYou consider yourself:\n    1. Friendly\n    2. Ambitious\n    3. Courageous\n    4. Creative\n    (Please choose by number.)')
    verification()

def question_three():
    if False:
        i = 10
        return i + 15
    print('\n*Third question*\nYou consider yourself:\n    1. Supportive\n    2. Prideful\n    3. Do-ers\n    4. Intuitive\n    (Please choose by number.)')
    verification()

def question_four():
    if False:
        while True:
            i = 10
    print('\n*Fourth question*\nYou consider yourself:\n    1. Loyal\n    2. Resourceful\n    3. Hands-on\n    4. Intelligent\n    (Please choose by number.)')
    verification()

def question_five():
    if False:
        return 10
    print('\n*Fifth question*\nYou consider yourself:\n    1. Generous\n    2. Confident\n    3. Conscientious\n    4. Witty\n    (Please choose by number.)')
    verification()

def question_opt():
    if False:
        while True:
            i = 10
    print('\n*Tie-breaker question*\nYou consider yourself:\n    1. Humble\n    2. Greedy\n    3. Instrospective\n    4. Reliable\n    (Please choose by number.)')
    verification()
print('================================================================================================')
print('\nYou will be selected for one of the houses depending on your choices. Be honest... if you wish.\n')
print('================================================================================================')
count = 0
while count < 6:
    count += 1
    if count == 1:
        question_one()
    elif count == 2:
        question_two()
    elif count == 3:
        question_three()
    elif count == 4:
        question_four()
    elif count == 5:
        question_five()
draw_times = 0

def draw_check():
    if False:
        i = 10
        return i + 15
    global draw_times
    draw_times += 1
    checker = Counter(houses)
    draw = checker.most_common(2)
    if draw_times <= 1:
        if draw[0][1] == draw[1][1]:
            print(f"You're between {draw[0][0]} and {draw[1][0]}. I'll make one last question to make the last decision.")
            question_opt()
            return draw_check()
    elif draw_times > 1:
        draw = checker.most_common(3)
        if draw[0][1] != draw[1][1]:
            print('\nBroke the deadlock. Making a decision now...\n')
            for i in range(0, 3):
                print('...\n')
                sleep(1)
            return max(draw, key=lambda points: draw[0])[0]
        print("\nWell, it've been hard. I'm going to pick a -semi- random house for you, since you are a special\n        wizard, and can't fit in only one house. I'm amazed.\n")
        for i in range(0, 3):
            print('...\n')
            sleep(1)
        final_house = choice(draw)
        return final_house[0]
    return
result = draw_check()
if not result:
    "\n    Obtain the house with the highest points based on the houses' values.\n    ! It will only print if there's no draw or tie-breaker.\n    "
    house = max(houses, key=lambda points: houses[points])
    print(f"\nYou've been selected for the {house} house. Congratulations.\n")
else:
    print(f"\nYou've been -hardly- selected for the {result} house. Congratulations... (not really).\n")