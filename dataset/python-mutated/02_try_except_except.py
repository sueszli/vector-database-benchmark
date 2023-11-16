def start_new_thread(function, args, kwargs={}):
    if False:
        i = 10
        return i + 15
    try:
        function()
    except SystemExit:
        pass
    except:
        args()

def interact():
    if False:
        print('Hello World!')
    while 1:
        try:
            more = 1
        except KeyboardInterrupt:
            more = 0