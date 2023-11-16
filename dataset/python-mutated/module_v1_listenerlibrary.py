class Listener:

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Listener2:

    def close(self):
        if False:
            return 10
        pass
ROBOT_LIBRARY_LISTENER = [Listener(), Listener2()]