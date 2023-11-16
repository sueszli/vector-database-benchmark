from urllib.request import urlopen
import time

def get_load_time(url):
    if False:
        print('Hello World!')
    'This function takes a user defined url as input\n    and returns the time taken to load that url in seconds.\n\n    Args:\n        url (string): The user defined url.\n\n    Returns:\n        time_to_load (float): The time taken to load the website in seconds.\n    '
    if ('https' or 'http') in url:
        open_this_url = urlopen(url)
    else:
        open_this_url = urlopen('https://' + url)
    start_time = time.time()
    open_this_url.read()
    end_time = time.time()
    open_this_url.close()
    time_to_load = end_time - start_time
    return time_to_load
if __name__ == '__main__':
    url = input('Enter the url whose loading time you want to check: ')
    print(f'\nThe time taken to load {url} is {get_load_time(url):.2} seconds.')