def keyword_in_my_lib_file():
    if False:
        return 10
    print('Here we go!!')

def embedded(arg):
    if False:
        while True:
            i = 10
    print(arg)
embedded.robot_name = 'Keyword with embedded ${arg} in MyLibFile'