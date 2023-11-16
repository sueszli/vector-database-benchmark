import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from googlesearch import search

def is_valid_url(url):
    if False:
        return 10
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def sanitize_url(url):
    if False:
        print('Hello World!')
    return urljoin(url, urlparse(url).path)

def check_local_file_access(url):
    if False:
        print('Hello World!')
    local_prefixes = ['file:///', 'file://localhost', 'http://localhost', 'https://localhost']
    return any((url.startswith(prefix) for prefix in local_prefixes))

def get_response(url, timeout=10) -> tuple:
    if False:
        print('Hello World!')
    '\n    Get the response from the URL.\n\n    Parameters:\n    ----------\n        url (str): The URL to get the response from.\n        timeout (int): The timeout for the HTTP request.\n\n    Returns:\n    -------\n        response (requests.models.Response): The response from the URL.\n        error (str): The error message if any.\n    '
    try:
        if check_local_file_access(url):
            raise ValueError('Access to local files is restricted')
        if not url.startswith('http://') and (not url.startswith('https://')):
            raise ValueError('Invalid URL format')
        sanitized_url = sanitize_url(url)
        user_agent_header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        response = requests.get(sanitized_url, headers=user_agent_header, timeout=timeout)
        if response.status_code >= 400:
            return (None, f'Error: HTTP {response.status_code} error')
        return (response, None)
    except ValueError as ve:
        return (None, f'Error: {str(ve)}')
    except requests.exceptions.RequestException as re:
        return (None, f'Error: {str(re)}')

def parse_web(url) -> str:
    if False:
        for i in range(10):
            print('nop')
    (response, potential_error) = get_response(url)
    if response is None:
        return potential_error
    if response.status_code >= 400:
        return f'Error: HTTP {str(response.status_code)} error'
    soup = BeautifulSoup(response.text, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    text = '\n'.join((chunk for chunk in chunks if chunk))
    return text

def google_search(keyword, num_results=5) -> dict:
    if False:
        while True:
            i = 10
    '\n    Search on Google and return the results.\n\n    Parameters:\n    ----------\n        keyword (str): The keyword to search on Google.\n        num_results (int): The number of results to return.\n\n    Returns:\n    -------\n        result (dict): The search results. Format: {"keyword": keyword, "search_result": {url, content}}}\n\n    '
    search_result = {url: parse_web(url) for url in search(keyword, tld='com', num=num_results, stop=num_results, pause=2)}
    return {'keyword': keyword, 'search_result': search_result}
if __name__ == '__main__':
    query = 'what is penetration testing?'
    for url in search(query, tld='com', num=5, stop=5, pause=2):
        print(url)
        web_content = parse_web(url)
        print(web_content)