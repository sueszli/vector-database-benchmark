def f():
    if False:
        print('Hello World!')
    title = 'fred'
    match title:
        case 'Weapon 1 Swap':
            title = 'Alt Weapon 1'
        case 'Weapon 2 Swap':
            title = 'Alt Weapon 2'
    return title
print(f())