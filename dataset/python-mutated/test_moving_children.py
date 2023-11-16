import random
import flet_core as ft

def test_moving_children():
    if False:
        print('Hello World!')
    c = ft.Stack()
    c._Control__uid = '0'
    for i in range(0, 10):
        c.controls.append(ft.Container())
        c.controls[i]._Control__uid = f'_{i}'
    index = []
    added_controls = []
    removed_controls = []
    commands = []
    c.build_update_commands(index, commands, added_controls, removed_controls, False)

    def replace_controls(c):
        if False:
            print('Hello World!')
        random.shuffle(c.controls)
        commands.clear()
        r = set()
        for ctrl in c.controls:
            r.add(ctrl._Control__uid)
        c.build_update_commands(index, commands, added_controls, removed_controls, False)
        for cmd in commands:
            if cmd.name == 'add':
                for sub_cmd in cmd.commands:
                    r.add(sub_cmd.attrs['id'])
            elif cmd.name == 'remove':
                for v in cmd.values:
                    r.remove(v)
        assert len(r) == len(c.controls)
    for i in range(0, 20):
        replace_controls(c)