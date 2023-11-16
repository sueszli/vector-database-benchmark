"""
There is a parking lot with only one empty spot. Given the initial state
of the parking lot and the final state. Each step we are only allowed to
move a car
out of its place and move it into the empty spot.
The goal is to find out the least movement needed to rearrange
the parking lot from the initial state to the final state.

Say the initial state is an array:

[1, 2, 3, 0, 4],
where 1, 2, 3, 4 are different cars, and 0 is the empty spot.

And the final state is

[0, 3, 2, 1, 4].
We can swap 1 with 0 in the initial array to get [0, 2, 3, 1, 4] and so on.
Each step swap with 0 only.

Edit:
Now also prints the sequence of changes in states.
Output of this example :-

initial: [1, 2, 3, 0, 4]
final:   [0, 3, 2, 1, 4]
Steps =  4
Sequence : 
0 2 3 1 4
2 0 3 1 4
2 3 0 1 4
0 3 2 1 4
"""

def garage(initial, final):
    if False:
        print('Hello World!')
    initial = initial[:]
    seq = []
    steps = 0
    while initial != final:
        zero = initial.index(0)
        if zero != final.index(0):
            car_to_move = final[zero]
            pos = initial.index(car_to_move)
            (initial[zero], initial[pos]) = (initial[pos], initial[zero])
        else:
            for i in range(len(initial)):
                if initial[i] != final[i]:
                    (initial[zero], initial[i]) = (initial[i], initial[zero])
                    break
        seq.append(initial[:])
        steps += 1
    return (steps, seq)
'\nthus:\n1 2 3 0 4 -- zero = 3, true, car_to_move = final[3] = 1,\n             pos = initial.index(1) = 0, switched [0], [3]\n0 2 3 1 4 -- zero = 0, f, initial[1] != final[1], switched 0,1\n2 0 3 1 4 -- zero = 1, t, car_to_move = final[1] = 3,\n             pos = initial.index(3) = 2, switched [1], [2]\n2 3 0 1 4 -- zero = 2, t, car_to_move = final[2] = 2, \n             pos = initial.index(2) = 0, switched [0], [2]\n0 3 2 1 4 -- initial == final\n'