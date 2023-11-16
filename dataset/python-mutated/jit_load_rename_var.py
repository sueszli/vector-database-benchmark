from paddle.base import unique_name
from paddle.base.dygraph.base import switch_to_static_graph

@switch_to_static_graph
def _generate_unique_var_name_sync_with_main_program(prefix):
    if False:
        while True:
            i = 10
    return unique_name.generate(prefix)

def rename_var_with_generator(names_old):
    if False:
        for i in range(10):
            print('nop')
    dict_rename_var_old_new = {}
    names_old = list(names_old)
    for (var_idx, name_old) in enumerate(names_old):
        while True:
            temp_name = name_old.split('_')
            if len(temp_name) > 1 and temp_name[-1].isnumeric():
                temp_name = '_'.join(temp_name[:-1])
            else:
                temp_name = '_'.join(temp_name)
            name_new = _generate_unique_var_name_sync_with_main_program(temp_name)
            if name_new not in names_old[:var_idx] + names_old[var_idx + 1:]:
                break
        dict_rename_var_old_new[name_old] = name_new
    return dict_rename_var_old_new