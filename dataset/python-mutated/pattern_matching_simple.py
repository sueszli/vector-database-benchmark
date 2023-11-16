match command.split():
    case [action, obj]:
        ...
match command.split():
    case [action]:
        ...
    case [action, obj]:
        ...
match command.split():
    case ['quit']:
        print('Goodbye!')
        quit_game()
    case ['look']:
        current_room.describe()
    case ['get', obj]:
        character.get(obj, current_room)
    case ['go', direction]:
        current_room = current_room.neighbor(direction)
match command.split():
    case ['drop', *objects]:
        for obj in objects:
            character.drop(obj, current_room)
match command.split():
    case ['quit']:
        pass
    case ['go', direction]:
        print('Going:', direction)
    case ['drop', *objects]:
        print('Dropping: ', *objects)
    case _:
        print(f"Sorry, I couldn't understand {command!r}")
match command.split():
    case ['north'] | ['go', 'north']:
        current_room = current_room.neighbor('north')
    case ['get', obj] | ['pick', 'up', obj] | ['pick', obj, 'up']:
        ...
match command.split():
    case ['go', 'north' | 'south' | 'east' | 'west']:
        current_room = current_room.neighbor(...)
match command.split():
    case ['go', 'north' | 'south' | 'east' | 'west' as direction]:
        current_room = current_room.neighbor(direction)
match command.split():
    case ['go', direction] if direction in current_room.exits:
        current_room = current_room.neighbor(direction)
    case ['go', _]:
        print("Sorry, you can't go that way")
match event.get():
    case Click(position=[x, y]):
        handle_click_at(x, y)
    case KeyPress(key_name='Q') | Quit():
        game.quit()
    case KeyPress(key_name='up arrow'):
        game.go_north()
    case KeyPress():
        pass
    case other_event:
        raise ValueError(f'Unrecognized event: {other_event}')
match event.get():
    case Click([x, y], button=Button.LEFT):
        handle_click_at(x, y)
    case Click():
        pass

def where_is(point):
    if False:
        i = 10
        return i + 15
    match point:
        case Point(x=0, y=0):
            print('Origin')
        case Point(x=0, y=y):
            print(f'Y={y}')
        case Point(x=x, y=0):
            print(f'X={x}')
        case Point():
            print('Somewhere else')
        case _:
            print('Not a point')