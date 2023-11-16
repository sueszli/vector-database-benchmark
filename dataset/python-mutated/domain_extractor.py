"""
Write a function that when given a URL as a string, parses out just the domain name and returns it as a string. 

Examples:
domain_name("http://github.com/SaadBenn") == "github" 
domain_name("http://www.zombie-bites.com") == "zombie-bites"
domain_name("https://www.cnet.com") == "cnet"

Note: The idea is not to use any built-in libraries such as re (regular expression) or urlparse except .split() built-in function
"""

def domain_name_1(url):
    if False:
        print('Hello World!')
    full_domain_name = url.split('//')[-1]
    actual_domain = full_domain_name.split('.')
    if len(actual_domain) > 2:
        return actual_domain[1]
    return actual_domain[0]

def domain_name_2(url):
    if False:
        i = 10
        return i + 15
    return url.split('//')[-1].split('www.')[-1].split('.')[0]