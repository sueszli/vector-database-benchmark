def fix_shift_amount_list(shift_amount_list):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(shift_amount_list[0], (int, float)):
        shift_amount_list = [shift_amount_list]
    return shift_amount_list

def fix_full_shape_list(full_shape_list):
    if False:
        return 10
    if full_shape_list is not None and isinstance(full_shape_list[0], (int, float)):
        full_shape_list = [full_shape_list]
    return full_shape_list