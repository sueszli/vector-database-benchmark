import threading

def thread_job():
    if False:
        while True:
            i = 10
    print('This is a thread of %s' % threading.current_thread())

def main():
    if False:
        return 10
    thread = threading.Thread(target=thread_job)
    thread.start()
if __name__ == '__main__':
    main()