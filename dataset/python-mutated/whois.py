from six.moves.urllib.parse import urlparse
from thefuck.utils import for_app

@for_app('whois', at_least=1)
def match(command):
    if False:
        print('Hello World!')
    "\n    What the `whois` command returns depends on the 'Whois server' it contacted\n    and is not consistent through different servers. But there can be only two\n    types of errors I can think of with `whois`:\n        - `whois https://en.wikipedia.org/` → `whois en.wikipedia.org`;\n        - `whois en.wikipedia.org` → `whois wikipedia.org`.\n    So we match any `whois` command and then:\n        - if there is a slash: keep only the FQDN;\n        - if there is no slash but there is a point: removes the left-most\n          subdomain.\n\n    We cannot either remove all subdomains because we cannot know which part is\n    the subdomains and which is the domain, consider:\n        - www.google.fr → subdomain: www, domain: 'google.fr';\n        - google.co.uk → subdomain: None, domain; 'google.co.uk'.\n    "
    return True

def get_new_command(command):
    if False:
        while True:
            i = 10
    url = command.script_parts[1]
    if '/' in command.script:
        return 'whois ' + urlparse(url).netloc
    elif '.' in command.script:
        path = urlparse(url).path.split('.')
        return ['whois ' + '.'.join(path[n:]) for n in range(1, len(path))]