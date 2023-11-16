import requests

def tracegeoIp(ip):
    if False:
        i = 10
        return i + 15
    try:
        if '127.0.0.1' == ip:
            ip = 'https://geoip-db.com/json/'
        else:
            ip = 'https://geoip-db.com/jsonp/' + ip
        result = requests.get(ip).json()
    except Exception as e:
        print(e)
        result = 'Error. Verify your network connection.'
    return result