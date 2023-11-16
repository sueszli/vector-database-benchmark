import PySimpleGUI as sg
'\n    PySimpleGUI is designed & authored in Python to take full advantage the awesome Python constructs & capabilities.\n    Layouts are represented as lists to PySimpleGUI. Lists are fundamental in Python and have a number of powerful\n    capabilities that PySimpleGUI exploits.\n       \n    Many times PySimpleGUI programs can benefit from using CODE to GENERATE your layouts.  This Demo illustrates\n    a number of ways of "building" a layout. Some work on 3.5 and up.  Some are basic and show concatenation of rows\n    to build up a layout.  Some utilize generators.\n    \n    These 8 "Constructs" or Design Patterns demonstrate numerous ways of "generating" or building your layouts\n    0 - A simple list comprehension to build a row of buttons\n    1 - A simple list comprehension to build a column of buttons\n    2 - Concatenation of rows within a layout\n    3 - Concatenation of 2 complete layouts [[ ]] + [[ ]] = [[ ]]\n    4 - Concatenation of elements to form a single row [ [] + [] + [] ] = [[ ]]\n    5 - Questionnaire - Using a double list comprehension to build both rows and columns in a single line of code\n    6 - Questionnaire - Unwinding the comprehensions into 2 for loops instead\n    7 - Using the * operator to unpack generated items onto a single row \n    8 - Multiple Choice Test - a practical use showing list comprehension and concatenated layout\n'
"\n    Construct #0 - List comprehension to generate a row of Buttons\n\n    Comprehensions are super-powers of Python.  In this example we're using a comprehension to create 4 buttons that\n    are all on the same row.\n"

def layout0():
    if False:
        print('Hello World!')
    layout = [[sg.Button(i) for i in range(4)]]
    window = sg.Window('Generated Layouts', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #1 - List comprehension to generate a Column of Buttons\n\n    More list super-power, this time used to build a series of buttons doing DOWN the window instead of across\n\n'

def layout1():
    if False:
        return 10
    layout = [[sg.Button(i)] for i in range(4)]
    window = sg.Window('Generated Layouts', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #2 - List comprehension to generate a row of Buttons and concatenation of more lines of elements\n\n    Comprehensions are super-powers of Python.  In this example we\'re using a comprehension to create 4 buttons that\n    are all on the same row, just like the previous example.\n    However, here, we want to not just have a row of buttons, we want have an OK button at the bottom.\n    To do this, you "add" the rest of the GUI layout onto the end of the generated part.\n    \n    Note - you can\'t end the layout line after the +. If you wanted to put the OK button on the next line, you need\n    to add a \\ at the end of the first line.\n    See next Construct on how to not use a \\ that also results in a VISUALLY similar to a norma layout\n'

def layout2():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Button(i) for i in range(4)]] + [[sg.OK()]]
    window = sg.Window('Generated Layouts', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
"\n    Construct # 3 - Adding together what appears to be 2 layouts\n    \n    Same as layout2, except that the OK button is put on another line without using a \\ so that the layout appears to\n    look like a normal, multiline layout without a \\ at the end\n    \n    Also shown is the OLD tried and true way, using layout.append.  You will see the append technique in many of the\n    Demo programs and probably elsewhere.  Hoping to remove these and instead use this more explicit method of +=.\n    \n    Using the + operator, as you've already seen, can be used in the middle of the layout.  A call to append you cannot\n    use this way because it modifies the layout list directly.\n"

def layout3():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Button(i) for i in range(4)]]
    layout += [[sg.OK()]]
    layout.append([sg.Cancel()])
    window = sg.Window('Generated Layouts', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
"\n    Construct 4 - Using + to place Elements on the same row\n    \n    If you want to put elements on the same row, you can simply add them together.  All that is happening is that the\n    items in one list are added to the items in another.  That's true for all these contructs using +\n"

def layout4():
    if False:
        print('Hello World!')
    layout = [[sg.Text('Enter some info')] + [sg.Input()] + [sg.Exit()]]
    window = sg.Window('Generated Layouts', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #5 - Simple "concatenation" of layouts\n    Layouts are lists of lists.  Some of the examples and demo programs use a .append method to add rows to layouts.\n    These will soono be replaced with this new technique.  It\'s so simple that I don\'t know why it took so long to\n    find it.\n    This layout uses list comprehensions heavily, and even uses 2 of them. So, if you find them confusing, skip down\n    to the next Construct and you\'ll see the same layout built except for loops are used rather than comprehensions\n    \n    The next 3 examples all use this same window that is layed out like this:\n        Each row is:\n    Text1, Text2, Radio1, Radio2, Radio3, Radio4, Radio5\n    Text1, Text2, Radio1, Radio2, Radio3, Radio4, Radio5\n    ...\n    \n    It shows, in particular, this handy bit of layout building, a += to add on additional rows.\n    layout =  [[stuff on row 1], [stuff on row 2]]\n    layout += [[stuff on row 3]]\n    \n    Works as long as the things you are adding together look like this [[ ]]  (the famous double bracket layouts of PSG)\n'

def layout5():
    if False:
        for i in range(10):
            print('nop')
    questions = ('Managing your day-to-day life', 'Coping with problems in your life?', 'Concentrating?', 'Get along with people in your family?', 'Get along with people outside your family?', 'Get along well in social situations?', 'Feel close to another person', 'Feel like you had someone to turn to if you needed help?', 'Felt confident in yourself?')
    layout = [[sg.Text(qnum + 1, size=(2, 2)), sg.Text(q, size=(30, 2))] + [sg.Radio('', group_id=qnum, size=(7, 2), key=(qnum, col)) for col in range(5)] for (qnum, q) in enumerate(questions)]
    layout += [[sg.OK()]]
    window = sg.Window('Computed Layout Questionnaire', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #6 - Computed layout without using list comprehensions\n    This layout is identical to Contruct #5.  The difference is that rather than use list comprehensions, this code\n    uses for loops.  Perhaps if you\'re a beginner this version makes more sense?\n\n    In this example we start with a "blank layout" [[ ]] and add onto it.\n\n    Works as long as the things you are adding together look like this [[ ]]  (the famous double bracket layouts of PSG)\n'

def layout6():
    if False:
        while True:
            i = 10
    questions = ('Managing your day-to-day life', 'Coping with problems in your life?', 'Concentrating?', 'Get along with people in your family?', 'Get along with people outside your family?', 'Get along well in social situations?', 'Feel close to another person', 'Feel like you had someone to turn to if you needed help?', 'Felt confident in yourself?')
    layout = [[]]
    for (qnum, question) in enumerate(questions):
        row_layout = [sg.Text(qnum + 1, size=(2, 2)), sg.Text(question, size=(30, 2))]
        for radio_num in range(5):
            row_layout += [sg.Radio('', group_id=qnum, size=(7, 2), key=(qnum, radio_num))]
        layout += [row_layout]
    layout += [[sg.OK()]]
    window = sg.Window('Computed Layout Questionnaire', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #7 - * operator and list comprehensions \n        Using the * operator from inside the layout\n        List comprehension inside the layout\n        Addition of rows to layouts\n        All in a single variable assignment\n        \n    NOTE - this particular code, using the * operator, will not work on Python 2 and think it was added in Python 3.5\n    \n    This code shows a bunch of questions with Radio Button choices\n    \n    There is a double-loop comprehension used.  One that loops through the questions (rows) and the other loops through\n    the Radio Button choices.\n    Thus each row is:\n    Text1, Text2, Radio1, Radio2, Radio3, Radio4, Radio5\n    Text1, Text2, Radio1, Radio2, Radio3, Radio4, Radio5\n    Text1, Text2, Radio1, Radio2, Radio3, Radio4, Radio5\n    \n    What the * operator is doing in these cases is expanding the list they are in front of into a SERIES of items\n    from the list... one after another, as if they are separated with comma.  It\'s a way of "unpacking" from within\n    a statement.\n    \n    The result is a beautifully compact way to make a layout, still using a layout variable, that consists of a\n    variable number of rows and a variable number of columns in each row.\n'

def layout7():
    if False:
        for i in range(10):
            print('nop')
    questions = ('Managing your day-to-day life', 'Coping with problems in your life?', 'Concentrating?', 'Get along with people in your family?', 'Get along with people outside your family?', 'Get along well in social situations?', 'Feel close to another person', 'Feel like you had someone to turn to if you needed help?', 'Felt confident in yourself?')
    layout = [[*[sg.Text(qnum + 1, size=(2, 2)), sg.Text(q, size=(30, 2))], *[sg.Radio('', group_id=qnum, size=(7, 2), key=(qnum, col)) for col in range(5)]] for (qnum, q) in enumerate(questions)] + [[sg.OK()]]
    window = sg.Window('Questionnaire', layout)
    (event, values) = window.read()
    print(event, values)
    window.close()
'\n    Construct #8 - Computed layout using list comprehension and concatenation\n    This layout shows one practical use, a multiple choice test.  It\'s been left parse as to focus on the generation\n    part of the program.  For example, default keys are used on the Radio elements.  In reality you would likely\n    use a tuple of the question number and the answer number.\n\n    In this example we start with a "Header" Text element and build from there.\n'

def layout8():
    if False:
        print('Hello World!')
    q_and_a = [['1. What is the thing that makes light in our solar system', ['A. The Moon', 'B. Jupiter', 'C. I dunno']], ['2. What is Pluto', ['A. The 9th planet', 'B. A dwarf-planet', 'C. The 8th planet', 'D. Goofies pet dog']], ['3. When did man step foot on the moon', ['A. 1969', 'B. 1960', 'C. 1970', 'D. 1869']]]
    layout = [[sg.Text('Astronomy Quiz #1', font='ANY 15', size=(30, 2))]]
    for qa in q_and_a:
        q = qa[0]
        a_list = qa[1]
        layout += [[sg.Text(q)]] + [[sg.Radio(a, group_id=q)] for a in a_list] + [[sg.Text('_' * 50)]]
    layout += [[sg.Button('Submit Answers', key='SUBMIT')]]
    window = sg.Window('Multiple Choice Test', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'SUBMIT'):
            break
    sg.popup('The answers submitted were', values)
    window.close()
layout0()