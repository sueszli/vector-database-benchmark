url = 'https//www.domain.com/page?key1=value1&key2=value2&key3=value3&key4=value4'

def get_parameter_values(url):
    if False:
        return 10
    return list((key_value.split('=')[1] for key_value in url.split('?')[1].split('&')))
print(get_parameter_values(url))