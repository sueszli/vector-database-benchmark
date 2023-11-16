"""
gitfiti

noun : Carefully crafted graffiti in a GitHub commit history calendar
"""
from datetime import datetime, timedelta
import itertools
import json
import math
import os
try:
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
except ImportError:
    from urllib2 import HTTPError, URLError, urlopen
try:
    raw_input
except NameError:
    raw_input = input
GITHUB_BASE_URL = 'https://github.com/'
FALLBACK_IMAGE = 'kitty'
TITLE = '\n          _ __  _____ __  _\n   ____ _(_) /_/ __(_) /_(_)\n  / __ `/ / __/ /_/ / __/ /\n / /_/ / / /_/ __/ / /_/ /\n \\__, /_/\\__/_/ /_/\\__/_/\n/____/\n'
KITTY = [[0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 4, 2, 4, 4, 4, 4, 2, 4, 0, 0], [0, 0, 4, 2, 2, 2, 2, 2, 2, 4, 0, 0], [2, 2, 4, 2, 4, 2, 2, 4, 2, 4, 2, 2], [0, 0, 4, 2, 2, 3, 3, 2, 2, 4, 0, 0], [2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2], [0, 0, 0, 3, 4, 4, 4, 4, 3, 0, 0, 0]]
ONEUP = [[0, 4, 4, 4, 4, 4, 4, 4, 0], [4, 3, 2, 2, 1, 2, 2, 3, 4], [4, 2, 2, 1, 1, 1, 2, 2, 4], [4, 3, 4, 4, 4, 4, 4, 3, 4], [4, 4, 1, 4, 1, 4, 1, 4, 4], [0, 4, 1, 1, 1, 1, 1, 4, 0], [0, 0, 4, 4, 4, 4, 4, 0, 0]]
ONEUP2 = [[0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0], [0, 4, 2, 2, 1, 1, 1, 2, 2, 4, 0], [4, 3, 2, 2, 1, 1, 1, 2, 2, 3, 4], [4, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4], [0, 4, 4, 1, 4, 1, 4, 1, 4, 4, 0], [0, 0, 4, 1, 1, 1, 1, 1, 4, 0, 0], [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0]]
HACKERSCHOOL = [[4, 4, 4, 4, 4, 4], [4, 3, 3, 3, 3, 4], [4, 1, 3, 3, 1, 4], [4, 3, 3, 3, 3, 4], [4, 4, 4, 4, 4, 4], [0, 0, 4, 4, 0, 0], [4, 4, 4, 4, 4, 4]]
OCTOCAT = [[0, 0, 0, 4, 0, 0, 0, 4, 0], [0, 0, 4, 4, 4, 4, 4, 4, 4], [0, 0, 4, 1, 3, 3, 3, 1, 4], [4, 0, 3, 4, 3, 3, 3, 4, 3], [0, 4, 0, 0, 4, 4, 4, 0, 0], [0, 0, 4, 4, 4, 4, 4, 4, 4], [0, 0, 4, 0, 4, 0, 4, 0, 4]]
OCTOCAT2 = [[0, 0, 4, 0, 0, 4, 0], [0, 4, 4, 4, 4, 4, 4], [0, 4, 1, 3, 3, 1, 4], [0, 4, 4, 4, 4, 4, 4], [4, 0, 0, 4, 4, 0, 0], [0, 4, 4, 4, 4, 4, 0], [0, 0, 0, 4, 4, 4, 0]]
HELLO = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 4], [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 4], [0, 3, 3, 3, 0, 2, 3, 3, 0, 3, 0, 3, 0, 1, 3, 1, 0, 3], [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 3], [0, 3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 2], [0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0], [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 4]]
HEART1 = [[0, 1, 1, 0, 1, 1, 0], [1, 3, 3, 1, 3, 3, 1], [1, 3, 4, 3, 4, 3, 1], [1, 3, 4, 4, 4, 3, 1], [0, 1, 3, 4, 3, 1, 0], [0, 0, 1, 3, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
HEART2 = [[0, 5, 5, 0, 5, 5, 0], [5, 3, 3, 5, 3, 3, 5], [5, 3, 1, 3, 1, 3, 5], [5, 3, 1, 1, 1, 3, 5], [0, 5, 3, 1, 3, 5, 0], [0, 0, 5, 3, 5, 0, 0], [0, 0, 0, 5, 0, 0, 0]]
HIREME = [[1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 0, 2, 0, 3, 3, 3, 0, 2, 3, 3, 0, 0, 3, 3, 0, 3, 0, 0, 2, 3, 3], [4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4], [3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 3, 3], [2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]]
BEER = [[0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0], [0, 0, 1, 1, 1, 1, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0], [0, 2, 2, 2, 2, 2, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0], [2, 0, 2, 2, 2, 2, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0], [2, 0, 2, 2, 2, 2, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0], [0, 2, 2, 2, 2, 2, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0], [0, 0, 2, 2, 2, 2, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 0, 0, 3, 0]]
GLIDERS = [[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0], [0, 4, 0, 4, 0, 0, 4, 4, 0, 0, 0, 4, 0, 0], [0, 0, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0]]
HEART = [[0, 4, 4, 0, 4, 4, 0], [4, 2, 2, 4, 2, 2, 4], [4, 2, 2, 2, 2, 2, 4], [4, 2, 2, 2, 2, 2, 4], [0, 4, 2, 2, 2, 4, 0], [0, 0, 4, 2, 4, 0, 0], [0, 0, 0, 4, 0, 0, 0]]
HEART_SHINY = [[0, 4, 4, 0, 4, 4, 0], [4, 2, 0, 4, 2, 2, 4], [4, 0, 2, 2, 2, 2, 4], [4, 2, 2, 2, 2, 2, 4], [0, 4, 2, 2, 2, 4, 0], [0, 0, 4, 2, 4, 0, 0], [0, 0, 0, 4, 0, 0, 0]]
ASCII_TO_NUMBER = {'_': 0, '_': 1, '~': 2, '=': 3, '*': 4}

def str_to_sprite(content):
    if False:
        return 10
    lines = content.split('\n')

    def is_empty_line(line):
        if False:
            for i in range(10):
                print('nop')
        return len(line) != 0
    lines = filter(is_empty_line, lines)
    split_lines = [list(line) for line in lines]
    for line in split_lines:
        for (index, char) in enumerate(line):
            line[index] = ASCII_TO_NUMBER.get(char, 0)
    return split_lines
ONEUP_STR = str_to_sprite('\n *******\n*=~~-~~=*\n*~~---~~*\n*=*****=*\n**-*-*-**\n *-----*\n  *****\n')
IMAGES = {'kitty': KITTY, 'oneup': ONEUP, 'oneup2': ONEUP2, 'hackerschool': HACKERSCHOOL, 'octocat': OCTOCAT, 'octocat2': OCTOCAT2, 'hello': HELLO, 'heart1': HEART1, 'heart2': HEART2, 'hireme': HIREME, 'oneup_str': ONEUP_STR, 'beer': BEER, 'gliders': GLIDERS, 'heart': HEART, 'heart_shiny': HEART_SHINY}
SHELLS = {'bash': 'sh', 'powershell': 'ps1'}

def load_images(img_names):
    if False:
        print('Hello World!')
    'loads user images from given file(s)'
    if img_names[0] == '':
        return {}
    for image_name in img_names:
        with open(image_name) as img:
            loaded_imgs = {}
            img_list = ''
            img_line = ' '
            name = img.readline().replace('\n', '')
            name = name[1:]
            while True:
                img_line = img.readline()
                if img_line == '':
                    break
                img_line.replace('\n', '')
                if img_line[0] == ':':
                    loaded_imgs[name] = json.loads(img_list)
                    name = img_line[1:]
                    img_list = ''
                else:
                    img_list += img_line
            loaded_imgs[name] = json.loads(img_list)
            return loaded_imgs

def retrieve_contributions_calendar(username, base_url):
    if False:
        i = 10
        return i + 15
    'retrieves the GitHub commit calendar data for a username'
    base_url = base_url + 'users/' + username
    try:
        url = base_url + '/contributions'
        page = urlopen(url)
    except (HTTPError, URLError) as e:
        print('There was a problem fetching data from {0}'.format(url))
        print(e)
        raise SystemExit
    return page.read().decode('utf-8')

def parse_contributions_calendar(contributions_calendar):
    if False:
        i = 10
        return i + 15
    'Yield daily counts extracted from the embedded contributions SVG.'
    for line in contributions_calendar.splitlines():
        if 'data-date=' in line:
            commit = line.split('>')[1].split()[0]
            if commit.isnumeric():
                yield int(commit)

def find_max_daily_commits(contributions_calendar):
    if False:
        while True:
            i = 10
    'finds the highest number of commits in one day'
    daily_counts = parse_contributions_calendar(contributions_calendar)
    return max(daily_counts, default=0)

def calculate_multiplier(max_commits):
    if False:
        for i in range(10):
            print('nop')
    'calculates a multiplier to scale GitHub colors to commit history'
    m = max_commits / 4.0
    if m == 0:
        return 1
    m = math.ceil(m)
    m = int(m)
    return m

def get_start_date():
    if False:
        print('Hello World!')
    'returns a datetime object for the first sunday after one year ago today\n    at 12:00 noon'
    today = datetime.today()
    date = datetime(today.year - 1, today.month, today.day, 12)
    weekday = datetime.weekday(date)
    while weekday < 6:
        date = date + timedelta(1)
        weekday = datetime.weekday(date)
    return date

def generate_next_dates(start_date, offset=0):
    if False:
        for i in range(10):
            print('nop')
    'generator that returns the next date, requires a datetime object as\n    input. The offset is in weeks'
    start = offset * 7
    for i in itertools.count(start):
        yield (start_date + timedelta(i))

def generate_values_in_date_order(image, multiplier=1):
    if False:
        while True:
            i = 10
    height = 7
    width = len(image[0])
    for w in range(width):
        for h in range(height):
            yield (image[h][w] * multiplier)

def commit(commitdate, shell):
    if False:
        while True:
            i = 10
    template_bash = 'GIT_AUTHOR_DATE={0} GIT_COMMITTER_DATE={1} git commit --allow-empty -m "gitfiti" > /dev/null\n'
    template_powershell = '$Env:GIT_AUTHOR_DATE="{0}"\n$Env:GIT_COMMITTER_DATE="{1}"\ngit commit --allow-empty -m "gitfiti" | Out-Null\n'
    template = template_bash if shell == 'bash' else template_powershell
    return template.format(commitdate.isoformat(), commitdate.isoformat())

def fake_it(image, start_date, username, repo, git_url, shell, offset=0, multiplier=1):
    if False:
        for i in range(10):
            print('nop')
    template_bash = '#!/usr/bin/env bash\nREPO={0}\ngit init $REPO\ncd $REPO\ntouch README.md\ngit add README.md\ntouch gitfiti\ngit add gitfiti\n{1}\ngit branch -M main\ngit remote add origin {2}:{3}/$REPO.git\ngit pull origin main\ngit push -u origin main\n'
    template_powershell = 'cd $PSScriptRoot\n$REPO="{0}"\ngit init $REPO\ncd $REPO\nNew-Item README.md -ItemType file | Out-Null\ngit add README.md\nNew-Item gitfiti -ItemType file | Out-Null\ngit add gitfiti\n{1}\ngit branch -M main\ngit remote add origin {2}:{3}/$REPO.git\ngit pull origin main\ngit push -u origin main\n'
    template = template_bash if shell == 'bash' else template_powershell
    strings = []
    for (value, date) in zip(generate_values_in_date_order(image, multiplier), generate_next_dates(start_date, offset)):
        for _ in range(value):
            strings.append(commit(date, shell))
    return template.format(repo, ''.join(strings), git_url, username)

def save(output, filename):
    if False:
        i = 10
        return i + 15
    'Saves the list to a given filename'
    with open(filename, 'w') as f:
        f.write(output)
    os.chmod(filename, 493)

def request_user_input(prompt='> '):
    if False:
        return 10
    'Request input from the user and return what has been entered.'
    return raw_input(prompt)

def main():
    if False:
        for i in range(10):
            print('nop')
    print(TITLE)
    ghe = request_user_input('Enter GitHub URL (leave blank to use {}): '.format(GITHUB_BASE_URL))
    username = request_user_input('Enter your GitHub username: ')
    git_base = ghe if ghe else GITHUB_BASE_URL
    contributions_calendar = retrieve_contributions_calendar(username, git_base)
    max_daily_commits = find_max_daily_commits(contributions_calendar)
    m = calculate_multiplier(max_daily_commits)
    repo = request_user_input('Enter the name of the repository to use by gitfiti: ')
    offset = request_user_input('Enter the number of weeks to offset the image (from the left): ')
    offset = int(offset) if offset.strip() else 0
    print('By default gitfiti.py matches the darkest pixel to the highest\nnumber of commits found in your GitHub commit/activity calendar,\n\nCurrently this is: {0} commits\n\nEnter the word "gitfiti" to exceed your max\n(this option generates WAY more commits)\nAny other input will cause the default matching behavior'.format(max_daily_commits))
    match = request_user_input()
    match = m if match == 'gitfiti' else 1
    print('Enter file(s) to load images from (blank if not applicable)')
    img_names = request_user_input().split(' ')
    loaded_images = load_images(img_names)
    images = dict(IMAGES, **loaded_images)
    print('Enter the image name to gitfiti')
    print('Images: ' + ', '.join(images.keys()))
    image = request_user_input()
    image_name_fallback = FALLBACK_IMAGE
    if not image:
        image = IMAGES[image_name_fallback]
    else:
        try:
            image = images[image]
        except:
            image = IMAGES[image_name_fallback]
    start_date = get_start_date()
    fake_it_multiplier = m * match
    if not ghe:
        git_url = 'git@github.com'
    else:
        git_url = request_user_input('Enter Git URL like git@site.github.com: ')
    shell = ''
    while shell not in SHELLS.keys():
        shell = request_user_input('Enter the target shell ({}): '.format(' or '.join(SHELLS.keys())))
    output = fake_it(image, start_date, username, repo, git_url, shell, offset, fake_it_multiplier)
    output_filename = 'gitfiti.{}'.format(SHELLS[shell])
    save(output, output_filename)
    print('{} saved.'.format(output_filename))
    print('Create a new(!) repo named {0} at {1} and run the script'.format(repo, git_base))
if __name__ == '__main__':
    main()