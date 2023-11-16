import urllib.request, urllib.parse, urllib.error

def _raw_urlretrieve(url, target_file, context=None):
    if False:
        while True:
            i = 10
    handle = urllib.request.urlopen(url)
    if handle.getcode() >= 300:
        raise IOError('HTTP Error ' + str(handle.getcode()))
    with open(target_file, 'wb') as output:
        while True:
            data = handle.read(1024 * 1024)
            if data:
                output.write(data)
            else:
                break

def urlretrieve(url, target_file):
    if False:
        i = 10
        return i + 15
    print('Downloading ' + url + ' to ' + target_file)
    _raw_urlretrieve(url, target_file)