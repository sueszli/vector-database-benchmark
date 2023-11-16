def plain_list_dict_args_function(plain, *arg_list, **arg_dict):
    if False:
        return 10
    print('plain', plain, 'arg_list', arg_list, 'arg_dict', arg_dict)

def plain_list_args_function(plain, *arg_list):
    if False:
        for i in range(10):
            print('nop')
    print(plain, arg_list)

def plain_dict_args_function(plain, **arg_dict):
    if False:
        for i in range(10):
            print('nop')
    print(plain, arg_dict)
print('Function with plain arg and varargs dict:')
plain_dict_args_function(1, a=2, b=3, c=4)
plain_dict_args_function(1)
print('Function with plain arg and varargs list:')
plain_list_args_function(1, 2, 3, 4)
plain_list_args_function(1)
print('Function with plain arg, varargs list and varargs dict:')
plain_list_dict_args_function(1, 2, z=3)
plain_list_dict_args_function(1, 2, 3)
plain_list_dict_args_function(1, a=2, b=3, c=4)

def list_dict_args_function(*arg_list, **arg_dict):
    if False:
        i = 10
        return i + 15
    print(arg_list, arg_dict)

def list_args_function(*arg_list):
    if False:
        i = 10
        return i + 15
    print(arg_list)

def dict_args_function(**arg_dict):
    if False:
        while True:
            i = 10
    print(arg_dict)
print('Function with plain arg and varargs dict:')
dict_args_function(a=2, b=3, c=4)
dict_args_function()
print('Function with plain arg and varargs list:')
list_args_function(2, 3, 4)
list_args_function()
print('Function with plain arg, varargs list and varargs dict:')
list_dict_args_function(2, z=3)
list_dict_args_function(2, 3)
list_dict_args_function(a=2, b=3, c=4)