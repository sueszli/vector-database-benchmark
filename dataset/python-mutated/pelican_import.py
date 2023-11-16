import argparse
import datetime
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from html import unescape
from urllib.error import URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import urlretrieve
from pelican.log import init
from pelican.settings import DEFAULT_CONFIG
from pelican.utils import SafeDatetime, slugify
logger = logging.getLogger(__name__)

def decode_wp_content(content, br=True):
    if False:
        while True:
            i = 10
    pre_tags = {}
    if content.strip() == '':
        return ''
    content += '\n'
    if '<pre' in content:
        pre_parts = content.split('</pre>')
        last_pre = pre_parts.pop()
        content = ''
        pre_index = 0
        for pre_part in pre_parts:
            start = pre_part.find('<pre')
            if start == -1:
                content = content + pre_part
                continue
            name = f'<pre wp-pre-tag-{pre_index}></pre>'
            pre_tags[name] = pre_part[start:] + '</pre>'
            content = content + pre_part[0:start] + name
            pre_index += 1
        content = content + last_pre
    content = re.sub('<br />\\s*<br />', '\n\n', content)
    allblocks = '(?:table|thead|tfoot|caption|col|colgroup|tbody|tr|td|th|div|dl|dd|dt|ul|ol|li|pre|select|option|form|map|area|blockquote|address|math|style|p|h[1-6]|hr|fieldset|noscript|samp|legend|section|article|aside|hgroup|header|footer|nav|figure|figcaption|details|menu|summary)'
    content = re.sub('(<' + allblocks + '[^>]*>)', '\n\\1', content)
    content = re.sub('(</' + allblocks + '>)', '\\1\n\n', content)
    if '<object' in content:
        content = re.sub('\\s*<param([^>]*)>\\s*', '<param\\1>', content)
        content = re.sub('\\s*</embed>\\s*', '</embed>', content)
    pgraphs = filter(lambda s: s != '', re.split('\\n\\s*\\n', content))
    content = ''
    for p in pgraphs:
        content = content + '<p>' + p.strip() + '</p>\n'
    content = re.sub('<p>\\s*</p>', '', content)
    content = re.sub('<p>([^<]+)</(div|address|form)>', '<p>\\1</p></\\2>', content)
    content = re.sub('<p>\\s*(</?' + allblocks + '[^>]*>)\\s*</p>', '\\1', content)
    content = re.sub('<p>(<li.*)</p>', '\\1', content)
    content = re.sub('<p><blockquote([^>]*)>', '<blockquote\\1><p>', content)
    content = content.replace('</blockquote></p>', '</p></blockquote>')
    content = re.sub('<p>\\s*(</?' + allblocks + '[^>]*>)', '\\1', content)
    content = re.sub('(</?' + allblocks + '[^>]*>)\\s*</p>', '\\1', content)
    if br:

        def _preserve_newline(match):
            if False:
                return 10
            return match.group(0).replace('\n', '<WPPreserveNewline />')
        content = re.sub('/<(script|style).*?<\\/\\\\1>/s', _preserve_newline, content)
        content = re.sub('(?<!<br />)\\s*\\n', '<br />\n', content)
        content = content.replace('<WPPreserveNewline />', '\n')
    content = re.sub('(</?' + allblocks + '[^>]*>)\\s*<br />', '\\1', content)
    content = re.sub('<br />(\\s*</?(?:p|li|div|dl|dd|dt|th|pre|td|ul|ol)[^>]*>)', '\\1', content)
    content = re.sub('\\n</p>', '</p>', content)
    if pre_tags:

        def _multi_replace(dic, string):
            if False:
                print('Hello World!')
            pattern = '|'.join(map(re.escape, dic.keys()))
            return re.sub(pattern, lambda m: dic[m.group()], string)
        content = _multi_replace(pre_tags, content)
    content = re.sub('\\[caption(?:.*?)(?:caption=\\"(.*?)\\")?\\]((?:\\<a(?:.*?)\\>)?(?:\\<img.*?\\>)(?:\\<\\/a\\>)?)\\s?(.*?)\\[\\/caption\\]', '<figure>\\n\\2\\n<figcaption>\\1\\3</figcaption>\\n</figure>', content)
    return content

def xml_to_soup(xml):
    if False:
        return 10
    'Opens an xml file'
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        error = 'Missing dependency "BeautifulSoup4" and "lxml" required to import XML files.'
        sys.exit(error)
    with open(xml, encoding='utf-8') as infile:
        xmlfile = infile.read()
    soup = BeautifulSoup(xmlfile, 'xml')
    return soup

def get_filename(post_name, post_id):
    if False:
        print('Hello World!')
    if post_name is None or post_name.isspace():
        return post_id
    else:
        return post_name

def wp2fields(xml, wp_custpost=False):
    if False:
        for i in range(10):
            print('nop')
    'Opens a wordpress XML file, and yield Pelican fields'
    soup = xml_to_soup(xml)
    items = soup.rss.channel.findAll('item')
    for item in items:
        if item.find('status').string in ['publish', 'draft']:
            try:
                title = unescape(item.title.contents[0])
            except IndexError:
                title = 'No title [%s]' % item.find('post_name').string
                logger.warning('Post "%s" is lacking a proper title', title)
            post_name = item.find('post_name').string
            post_id = item.find('post_id').string
            filename = get_filename(post_name, post_id)
            content = item.find('encoded').string
            raw_date = item.find('post_date').string
            if raw_date == '0000-00-00 00:00:00':
                date = None
            else:
                date_object = SafeDatetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S')
                date = date_object.strftime('%Y-%m-%d %H:%M')
            author = item.find('creator').string
            categories = [cat.string for cat in item.findAll('category', {'domain': 'category'})]
            tags = [tag.string for tag in item.findAll('category', {'domain': 'post_tag'})]
            status = 'published' if item.find('status').string == 'publish' else item.find('status').string
            kind = 'article'
            post_type = item.find('post_type').string
            if post_type == 'page':
                kind = 'page'
            elif wp_custpost:
                if post_type == 'post':
                    pass
                elif post_type == 'attachment':
                    pass
                else:
                    kind = post_type
            yield (title, content, filename, date, author, categories, tags, status, kind, 'wp-html')

def blogger2fields(xml):
    if False:
        print('Hello World!')
    'Opens a blogger XML file, and yield Pelican fields'
    soup = xml_to_soup(xml)
    entries = soup.feed.findAll('entry')
    for entry in entries:
        raw_kind = entry.find('category', {'scheme': 'http://schemas.google.com/g/2005#kind'}).get('term')
        if raw_kind == 'http://schemas.google.com/blogger/2008/kind#post':
            kind = 'article'
        elif raw_kind == 'http://schemas.google.com/blogger/2008/kind#comment':
            kind = 'comment'
        elif raw_kind == 'http://schemas.google.com/blogger/2008/kind#page':
            kind = 'page'
        else:
            continue
        try:
            assert kind != 'comment'
            filename = entry.find('link', {'rel': 'alternate'})['href']
            filename = os.path.splitext(os.path.basename(filename))[0]
        except (AssertionError, TypeError, KeyError):
            filename = entry.find('id').string.split('.')[-1]
        title = entry.find('title').string or ''
        content = entry.find('content').string
        raw_date = entry.find('published').string
        if hasattr(SafeDatetime, 'fromisoformat'):
            date_object = SafeDatetime.fromisoformat(raw_date)
        else:
            date_object = SafeDatetime.strptime(raw_date[:23], '%Y-%m-%dT%H:%M:%S.%f')
        date = date_object.strftime('%Y-%m-%d %H:%M')
        author = entry.find('author').find('name').string
        tags = [tag.get('term') for tag in entry.findAll('category', {'scheme': 'http://www.blogger.com/atom/ns#'})]
        status = 'published'
        try:
            if entry.find('control').find('draft').string == 'yes':
                status = 'draft'
        except AttributeError:
            pass
        yield (title, content, filename, date, author, None, tags, status, kind, 'html')

def dc2fields(file):
    if False:
        print('Hello World!')
    'Opens a Dotclear export file, and yield pelican fields'
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        error = 'Missing dependency "BeautifulSoup4" and "lxml" required to import Dotclear files.'
        sys.exit(error)
    in_cat = False
    in_post = False
    category_list = {}
    posts = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line[:-1]
            if line.startswith('[category'):
                in_cat = True
            elif line.startswith('[post'):
                in_post = True
            elif in_cat:
                fields = line.split('","')
                if not line:
                    in_cat = False
                else:
                    fields[0] = fields[0][1:]
                    category_list[fields[0]] = fields[2]
            elif in_post:
                if not line:
                    in_post = False
                    break
                else:
                    posts.append(line)
    print('%i posts read.' % len(posts))
    subs = DEFAULT_CONFIG['SLUG_REGEX_SUBSTITUTIONS']
    for post in posts:
        fields = post.split('","')
        cat_id = fields[3]
        post_creadt = fields[6]
        post_format = fields[10]
        post_title = fields[13]
        post_excerpt = fields[14]
        post_excerpt_xhtml = fields[15]
        post_content = fields[16]
        post_content_xhtml = fields[17]
        post_meta = fields[27]
        post_creadt = ':'.join(post_creadt.split(':')[0:2])
        author = ''
        categories = []
        tags = []
        if cat_id:
            categories = [category_list[id].strip() for id in cat_id.split(',')]
        tag = post_meta.replace('{', '').replace('}', '').replace('a:1:s:3:\\"tag\\";a:', '').replace('a:0:', '')
        if len(tag) > 1:
            if int(len(tag[:1])) == 1:
                newtag = tag.split('"')[1]
                tags.append(BeautifulSoup(newtag, 'xml').decode('utf-8'))
            else:
                i = 1
                j = 1
                while i <= int(tag[:1]):
                    newtag = tag.split('"')[j].replace('\\', '')
                    tags.append(BeautifulSoup(newtag, 'xml').decode('utf-8'))
                    i = i + 1
                    if j < int(tag[:1]) * 2:
                        j = j + 2
        '\n        dotclear2 does not use markdown by default unless\n        you use the markdown plugin\n        Ref: http://plugins.dotaddict.org/dc2/details/formatting-markdown\n        '
        if post_format == 'markdown':
            content = post_excerpt + post_content
        else:
            content = post_excerpt_xhtml + post_content_xhtml
            content = content.replace('\\n', '')
            post_format = 'html'
        kind = 'article'
        status = 'published'
        yield (post_title, content, slugify(post_title, regex_subs=subs), post_creadt, author, categories, tags, status, kind, post_format)

def _get_tumblr_posts(api_key, blogname, offset=0):
    if False:
        print('Hello World!')
    import json
    import urllib.request as urllib_request
    url = 'https://api.tumblr.com/v2/blog/%s.tumblr.com/posts?api_key=%s&offset=%d&filter=raw' % (blogname, api_key, offset)
    request = urllib_request.Request(url)
    handle = urllib_request.urlopen(request)
    posts = json.loads(handle.read().decode('utf-8'))
    return posts.get('response').get('posts')

def tumblr2fields(api_key, blogname):
    if False:
        i = 10
        return i + 15
    'Imports Tumblr posts (API v2)'
    offset = 0
    posts = _get_tumblr_posts(api_key, blogname, offset)
    subs = DEFAULT_CONFIG['SLUG_REGEX_SUBSTITUTIONS']
    while len(posts) > 0:
        for post in posts:
            title = post.get('title') or post.get('source_title') or post.get('type').capitalize()
            slug = post.get('slug') or slugify(title, regex_subs=subs)
            tags = post.get('tags')
            timestamp = post.get('timestamp')
            date = SafeDatetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            slug = SafeDatetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc).strftime('%Y-%m-%d-') + slug
            format = post.get('format')
            content = post.get('body')
            type = post.get('type')
            if type == 'photo':
                if format == 'markdown':
                    fmtstr = '![%s](%s)'
                else:
                    fmtstr = '<img alt="%s" src="%s" />'
                content = '\n'.join((fmtstr % (photo.get('caption'), photo.get('original_size').get('url')) for photo in post.get('photos')))
            elif type == 'quote':
                if format == 'markdown':
                    fmtstr = '\n\n&mdash; %s'
                else:
                    fmtstr = '<p>&mdash; %s</p>'
                content = post.get('text') + fmtstr % post.get('source')
            elif type == 'link':
                if format == 'markdown':
                    fmtstr = '[via](%s)\n\n'
                else:
                    fmtstr = '<p><a href="%s">via</a></p>\n'
                content = fmtstr % post.get('url') + post.get('description')
            elif type == 'audio':
                if format == 'markdown':
                    fmtstr = '[via](%s)\n\n'
                else:
                    fmtstr = '<p><a href="%s">via</a></p>\n'
                content = fmtstr % post.get('source_url') + post.get('caption') + post.get('player')
            elif type == 'video':
                if format == 'markdown':
                    fmtstr = '[via](%s)\n\n'
                else:
                    fmtstr = '<p><a href="%s">via</a></p>\n'
                source = fmtstr % post.get('source_url')
                caption = post.get('caption')
                players = [player.get('embed_code') or None for player in post.get('player')]
                if len(players) > 0 and all((player is None for player in players)):
                    players = "<p>(This video isn't available anymore.)</p>\n"
                else:
                    players = '\n'.join(players)
                content = source + caption + players
            elif type == 'answer':
                title = post.get('question')
                content = '<p><a href="%s" rel="external nofollow">%s</a>: %s</p>\n %s' % (post.get('asking_name'), post.get('asking_url'), post.get('question'), post.get('answer'))
            content = content.rstrip() + '\n'
            kind = 'article'
            status = 'published'
            yield (title, content, slug, date, post.get('blog_name'), [type], tags, status, kind, format)
        offset += len(posts)
        posts = _get_tumblr_posts(api_key, blogname, offset)

def feed2fields(file):
    if False:
        for i in range(10):
            print('nop')
    'Read a feed and yield pelican fields'
    import feedparser
    d = feedparser.parse(file)
    subs = DEFAULT_CONFIG['SLUG_REGEX_SUBSTITUTIONS']
    for entry in d.entries:
        date = time.strftime('%Y-%m-%d %H:%M', entry.updated_parsed) if hasattr(entry, 'updated_parsed') else None
        author = entry.author if hasattr(entry, 'author') else None
        tags = [e['term'] for e in entry.tags] if hasattr(entry, 'tags') else None
        slug = slugify(entry.title, regex_subs=subs)
        kind = 'article'
        yield (entry.title, entry.description, slug, date, author, [], tags, None, kind, 'html')

def build_header(title, date, author, categories, tags, slug, status=None, attachments=None):
    if False:
        print('Hello World!')
    'Build a header from a list of fields'
    from docutils.utils import column_width
    header = '{}\n{}\n'.format(title, '#' * column_width(title))
    if date:
        header += ':date: %s\n' % date
    if author:
        header += ':author: %s\n' % author
    if categories:
        header += ':category: %s\n' % ', '.join(categories)
    if tags:
        header += ':tags: %s\n' % ', '.join(tags)
    if slug:
        header += ':slug: %s\n' % slug
    if status:
        header += ':status: %s\n' % status
    if attachments:
        header += ':attachments: %s\n' % ', '.join(attachments)
    header += '\n'
    return header

def build_asciidoc_header(title, date, author, categories, tags, slug, status=None, attachments=None):
    if False:
        for i in range(10):
            print('nop')
    'Build a header from a list of fields'
    header = '= %s\n' % title
    if author:
        header += '%s\n' % author
        if date:
            header += '%s\n' % date
    if categories:
        header += ':category: %s\n' % ', '.join(categories)
    if tags:
        header += ':tags: %s\n' % ', '.join(tags)
    if slug:
        header += ':slug: %s\n' % slug
    if status:
        header += ':status: %s\n' % status
    if attachments:
        header += ':attachments: %s\n' % ', '.join(attachments)
    header += '\n'
    return header

def build_markdown_header(title, date, author, categories, tags, slug, status=None, attachments=None):
    if False:
        for i in range(10):
            print('nop')
    'Build a header from a list of fields'
    header = 'Title: %s\n' % title
    if date:
        header += 'Date: %s\n' % date
    if author:
        header += 'Author: %s\n' % author
    if categories:
        header += 'Category: %s\n' % ', '.join(categories)
    if tags:
        header += 'Tags: %s\n' % ', '.join(tags)
    if slug:
        header += 'Slug: %s\n' % slug
    if status:
        header += 'Status: %s\n' % status
    if attachments:
        header += 'Attachments: %s\n' % ', '.join(attachments)
    header += '\n'
    return header

def get_ext(out_markup, in_markup='html'):
    if False:
        i = 10
        return i + 15
    if out_markup == 'asciidoc':
        ext = '.adoc'
    elif in_markup == 'markdown' or out_markup == 'markdown':
        ext = '.md'
    else:
        ext = '.rst'
    return ext

def get_out_filename(output_path, filename, ext, kind, dirpage, dircat, categories, wp_custpost, slug_subs):
    if False:
        return 10
    filename = os.path.basename(filename)
    filename = re.sub('[<>:"/\\\\|?*^% ]', '-', filename)
    filename = filename.lstrip('.')
    if not filename:
        filename = '_'
    filename = filename[:249]
    out_filename = os.path.join(output_path, filename + ext)
    if dirpage and kind == 'page':
        pages_dir = os.path.join(output_path, 'pages')
        if not os.path.isdir(pages_dir):
            os.mkdir(pages_dir)
        out_filename = os.path.join(pages_dir, filename + ext)
    elif not dirpage and kind == 'page':
        pass
    elif kind != 'article':
        if wp_custpost:
            typename = slugify(kind, regex_subs=slug_subs)
        else:
            typename = ''
            kind = 'article'
        if dircat and len(categories) > 0:
            catname = slugify(categories[0], regex_subs=slug_subs, preserve_case=True)
        else:
            catname = ''
        out_filename = os.path.join(output_path, typename, catname, filename + ext)
        if not os.path.isdir(os.path.join(output_path, typename, catname)):
            os.makedirs(os.path.join(output_path, typename, catname))
    elif dircat and len(categories) > 0:
        catname = slugify(categories[0], regex_subs=slug_subs, preserve_case=True)
        out_filename = os.path.join(output_path, catname, filename + ext)
        if not os.path.isdir(os.path.join(output_path, catname)):
            os.mkdir(os.path.join(output_path, catname))
    return out_filename

def get_attachments(xml):
    if False:
        for i in range(10):
            print('nop')
    'returns a dictionary of posts that have attachments with a list\n    of the attachment_urls\n    '
    soup = xml_to_soup(xml)
    items = soup.rss.channel.findAll('item')
    names = {}
    attachments = []
    for item in items:
        kind = item.find('post_type').string
        post_name = item.find('post_name').string
        post_id = item.find('post_id').string
        if kind == 'attachment':
            attachments.append((item.find('post_parent').string, item.find('attachment_url').string))
        else:
            filename = get_filename(post_name, post_id)
            names[post_id] = filename
    attachedposts = defaultdict(set)
    for (parent, url) in attachments:
        try:
            parent_name = names[parent]
        except KeyError:
            parent_name = None
        attachedposts[parent_name].add(url)
    return attachedposts

def download_attachments(output_path, urls):
    if False:
        for i in range(10):
            print('nop')
    'Downloads WordPress attachments and returns a list of paths to\n    attachments that can be associated with a post (relative path to output\n    directory). Files that fail to download, will not be added to posts'
    locations = {}
    for url in urls:
        path = urlparse(url).path
        path = path.split('/')
        filename = path.pop(-1)
        localpath = ''
        for item in path:
            if sys.platform != 'win32' or ':' not in item:
                localpath = os.path.join(localpath, item)
        full_path = os.path.join(output_path, localpath)
        (scheme, netloc, path, query, fragment) = urlsplit(url)
        if scheme != 'file':
            path = quote(path)
            url = urlunsplit((scheme, netloc, path, query, fragment))
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        print(f'downloading {filename}')
        try:
            urlretrieve(url, os.path.join(full_path, filename))
            locations[url] = os.path.join(localpath, filename)
        except (URLError, OSError) as e:
            logger.warning('No file could be downloaded from %s\n%s', url, e)
    return locations

def is_pandoc_needed(in_markup):
    if False:
        for i in range(10):
            print('nop')
    return in_markup in ('html', 'wp-html')

def get_pandoc_version():
    if False:
        return 10
    cmd = ['pandoc', '--version']
    try:
        output = subprocess.check_output(cmd, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        logger.warning('Pandoc version unknown: %s', e)
        return ()
    return tuple((int(i) for i in output.split()[1].split('.')))

def update_links_to_attached_files(content, attachments):
    if False:
        i = 10
        return i + 15
    for (old_url, new_path) in attachments.items():
        http_url = old_url.replace('https://', 'http://')
        https_url = old_url.replace('http://', 'https://')
        for url in [http_url, https_url]:
            content = content.replace(url, '{static}' + new_path)
    return content

def fields2pelican(fields, out_markup, output_path, dircat=False, strip_raw=False, disable_slugs=False, dirpage=False, filename_template=None, filter_author=None, wp_custpost=False, wp_attach=False, attachments=None):
    if False:
        while True:
            i = 10
    pandoc_version = get_pandoc_version()
    posts_require_pandoc = []
    slug_subs = DEFAULT_CONFIG['SLUG_REGEX_SUBSTITUTIONS']
    for (title, content, filename, date, author, categories, tags, status, kind, in_markup) in fields:
        if filter_author and filter_author != author:
            continue
        if is_pandoc_needed(in_markup) and (not pandoc_version):
            posts_require_pandoc.append(filename)
        slug = not disable_slugs and filename or None
        if wp_attach and attachments:
            try:
                urls = attachments[filename]
                links = download_attachments(output_path, urls)
            except KeyError:
                links = None
        else:
            links = None
        ext = get_ext(out_markup, in_markup)
        if ext == '.adoc':
            header = build_asciidoc_header(title, date, author, categories, tags, slug, status, attachments)
        elif ext == '.md':
            header = build_markdown_header(title, date, author, categories, tags, slug, status, links.values() if links else None)
        else:
            out_markup = 'rst'
            header = build_header(title, date, author, categories, tags, slug, status, links.values() if links else None)
        out_filename = get_out_filename(output_path, filename, ext, kind, dirpage, dircat, categories, wp_custpost, slug_subs)
        print(out_filename)
        if in_markup in ('html', 'wp-html'):
            with tempfile.TemporaryDirectory() as tmpdir:
                html_filename = os.path.join(tmpdir, 'pandoc-input.html')
                if in_markup == 'wp-html':
                    new_content = decode_wp_content(content)
                else:
                    paragraphs = content.splitlines()
                    paragraphs = [f'<p>{p}</p>' for p in paragraphs]
                    new_content = ''.join(paragraphs)
                with open(html_filename, 'w', encoding='utf-8') as fp:
                    fp.write(new_content)
                if pandoc_version < (2,):
                    parse_raw = '--parse-raw' if not strip_raw else ''
                    wrap_none = '--wrap=none' if pandoc_version >= (1, 16) else '--no-wrap'
                    cmd = 'pandoc --normalize {0} --from=html --to={1} {2} -o "{3}" "{4}"'
                    cmd = cmd.format(parse_raw, out_markup if out_markup != 'markdown' else 'gfm', wrap_none, out_filename, html_filename)
                else:
                    from_arg = '-f html+raw_html' if not strip_raw else '-f html'
                    cmd = 'pandoc {0} --to={1}-smart --wrap=none -o "{2}" "{3}"'
                    cmd = cmd.format(from_arg, out_markup if out_markup != 'markdown' else 'gfm', out_filename, html_filename)
                try:
                    rc = subprocess.call(cmd, shell=True)
                    if rc < 0:
                        error = 'Child was terminated by signal %d' % -rc
                        exit(error)
                    elif rc > 0:
                        error = 'Please, check your Pandoc installation.'
                        exit(error)
                except OSError as e:
                    error = 'Pandoc execution failed: %s' % e
                    exit(error)
            with open(out_filename, encoding='utf-8') as fs:
                content = fs.read()
                if out_markup == 'markdown':
                    content = content.replace('\\\n ', '  \n')
                    content = content.replace('\\\n', '  \n')
            if wp_attach and links:
                content = update_links_to_attached_files(content, links)
        with open(out_filename, 'w', encoding='utf-8') as fs:
            fs.write(header + content)
    if posts_require_pandoc:
        logger.error('Pandoc must be installed to import the following posts:\n  {}'.format('\n  '.join(posts_require_pandoc)))
    if wp_attach and attachments and (None in attachments):
        print("downloading attachments that don't have a parent post")
        urls = attachments[None]
        download_attachments(output_path, urls)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Transform feed, Blogger, Dotclear, Tumblr, or WordPress files into reST (rst) or Markdown (md) files. Be sure to have pandoc installed.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='input', help='The input file to read')
    parser.add_argument('--blogger', action='store_true', dest='blogger', help='Blogger XML export')
    parser.add_argument('--dotclear', action='store_true', dest='dotclear', help='Dotclear export')
    parser.add_argument('--tumblr', action='store_true', dest='tumblr', help='Tumblr export')
    parser.add_argument('--wpfile', action='store_true', dest='wpfile', help='Wordpress XML export')
    parser.add_argument('--feed', action='store_true', dest='feed', help='Feed to parse')
    parser.add_argument('-o', '--output', dest='output', default='content', help='Output path')
    parser.add_argument('-m', '--markup', dest='markup', default='rst', help='Output markup format (supports rst & markdown)')
    parser.add_argument('--dir-cat', action='store_true', dest='dircat', help='Put files in directories with categories name')
    parser.add_argument('--dir-page', action='store_true', dest='dirpage', help='Put files recognised as pages in "pages/" sub-directory (blogger and wordpress import only)')
    parser.add_argument('--filter-author', dest='author', help='Import only post from the specified author')
    parser.add_argument('--strip-raw', action='store_true', dest='strip_raw', help="Strip raw HTML code that can't be converted to markup such as flash embeds or iframes (wordpress import only)")
    parser.add_argument('--wp-custpost', action='store_true', dest='wp_custpost', help='Put wordpress custom post types in directories. If used with --dir-cat option directories will be created as /post_type/category/ (wordpress import only)')
    parser.add_argument('--wp-attach', action='store_true', dest='wp_attach', help="(wordpress import only) Download files uploaded to wordpress as attachments. Files will be added to posts as a list in the post header. All files will be downloaded, even if they aren't associated with a post. Files will be downloaded with their original path inside the output directory. e.g. output/wp-uploads/date/postname/file.jpg -- Requires an internet connection --")
    parser.add_argument('--disable-slugs', action='store_true', dest='disable_slugs', help='Disable storing slugs from imported posts within output. With this disabled, your Pelican URLs may not be consistent with your original posts.')
    parser.add_argument('-b', '--blogname', dest='blogname', help='Blog name (Tumblr import only)')
    args = parser.parse_args()
    input_type = None
    if args.blogger:
        input_type = 'blogger'
    elif args.dotclear:
        input_type = 'dotclear'
    elif args.tumblr:
        input_type = 'tumblr'
    elif args.wpfile:
        input_type = 'wordpress'
    elif args.feed:
        input_type = 'feed'
    else:
        error = 'You must provide either --blogger, --dotclear, --tumblr, --wpfile or --feed options'
        exit(error)
    if not os.path.exists(args.output):
        try:
            os.mkdir(args.output)
        except OSError:
            error = 'Unable to create the output folder: ' + args.output
            exit(error)
    if args.wp_attach and input_type != 'wordpress':
        error = 'You must be importing a wordpress xml to use the --wp-attach option'
        exit(error)
    if input_type == 'blogger':
        fields = blogger2fields(args.input)
    elif input_type == 'dotclear':
        fields = dc2fields(args.input)
    elif input_type == 'tumblr':
        fields = tumblr2fields(args.input, args.blogname)
    elif input_type == 'wordpress':
        fields = wp2fields(args.input, args.wp_custpost or False)
    elif input_type == 'feed':
        fields = feed2fields(args.input)
    if args.wp_attach:
        attachments = get_attachments(args.input)
    else:
        attachments = None
    init()
    fields2pelican(fields, args.markup, args.output, dircat=args.dircat or False, dirpage=args.dirpage or False, strip_raw=args.strip_raw or False, disable_slugs=args.disable_slugs or False, filter_author=args.author, wp_custpost=args.wp_custpost or False, wp_attach=args.wp_attach or False, attachments=attachments or None)