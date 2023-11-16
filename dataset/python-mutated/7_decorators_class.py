from functools import wraps

class logit(object):

    def __init__(self, logfile='out.log'):
        if False:
            return 10
        self.logfile = logfile

    def __call__(self, func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            if False:
                return 10
            log_string = func.__name__ + ' was called'
            print(log_string)
            with open(self.logfile, 'a') as opened_file:
                opened_file.write(log_string + '\n')
            self.notify()
            return func(*args, **kwargs)
        return wrapped_function

    def notify(self):
        if False:
            while True:
                i = 10
        pass

class email_logit(logit):

    def __init__(self, email='admin@myproject.com', *args, **kwargs):
        if False:
            print('Hello World!')
        self.email = email
        super(logit, self).__init__(*args, **kwargs)

    def notify(self):
        if False:
            print('Hello World!')
        pass