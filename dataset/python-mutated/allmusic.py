import difflib
import re

def allmusic_artistfind(data, artist):
    if False:
        return 10
    data = data.decode('utf-8')
    artists = []
    artistlist = re.findall('class="artist">\\s*(.*?)\\s*</li', data, re.S)
    for item in artistlist:
        artistdata = {}
        artistname = re.search('class="name">.*?>(.*?)</a', item, re.S)
        if artistname:
            artistdata['artist'] = artistname.group(1)
        else:
            continue
        artistmatch = difflib.SequenceMatcher(None, artist.lower(), artistdata['artist'].lower()).ratio()
        if artistmatch > 0.95:
            artisturl = re.search('class="name">\\s*<a href="(.*?)"', item)
            if artisturl:
                artistdata['url'] = artisturl.group(1)
            else:
                continue
            artists.append(artistdata)
            break
    return artists

def allmusic_artistdetails(data):
    if False:
        return 10
    data = data.decode('utf-8')
    artistdata = {}
    artist = re.search('artist-name" itemprop="name">\\s*(.*?)\\s*<', data)
    if artist:
        artistdata['artist'] = artist.group(1)
    else:
        return
    active = re.search('class="active-dates">.*?<div>(.*?)<', data, re.S)
    if active:
        artistdata['active'] = active.group(1)
    begin = re.search('class="birth">.*?<h4>\\s*(.*?)\\s*<', data, re.S)
    if begin and begin.group(1) == 'Born':
        born = re.search('class="birth">.*?<a.*?>(.*?)<', data, re.S)
        if born:
            artistdata['born'] = born.group(1)
    elif begin and begin.group(1) == 'Formed':
        formed = re.search('class="birth">.*?<a.*?>(.*?)<', data, re.S)
        if formed:
            artistdata['formed'] = formed.group(1)
    end = re.search('class="died">.*?<h4>\\s*(.*?)\\s*<', data, re.S)
    if end and end.group(1) == 'Died':
        died = re.search('class="died">.*?<a.*?>(.*?)<', data, re.S)
        if died:
            artistdata['died'] = died.group(1)
    elif end and end.group(1) == 'Disbanded':
        disbanded = re.search('class="died">.*?<a.*?>(.*?)<', data, re.S)
        if disbanded:
            artistdata['disbanded'] = disbanded.group(1)
    genre = re.search('class="genre">.*?<a.*?>(.*?)<', data, re.S)
    if genre:
        artistdata['genre'] = genre.group(1)
    styledata = re.search('class="styles">.*?<div>\\s*(.*?)\\s*</div', data, re.S)
    if styledata:
        styles = re.findall('">(.*?)<', styledata.group(1))
        if styles:
            artistdata['styles'] = ' / '.join(styles)
    mooddata = re.search('class="moods">.*?<li>\\s*(.*?)\\s*</ul', data, re.S)
    if mooddata:
        moods = re.findall('">(.*?)<', mooddata.group(1))
        if moods:
            artistdata['moods'] = ' / '.join(moods)
    thumbsdata = re.search('class="artist-image">.*?<img src="(.*?)"', data, re.S)
    if thumbsdata:
        thumbs = []
        thumbdata = {}
        thumb = thumbsdata.group(1).rstrip('?partner=allrovi.com')
        if thumb.endswith('f=4'):
            thumbdata['image'] = thumb.replace('f=4', 'f=0')
            thumbdata['preview'] = thumb.replace('f=4', 'f=2')
        else:
            thumbdata['image'] = thumb
            thumbdata['preview'] = thumb
        thumbdata['aspect'] = 'thumb'
        thumbs.append(thumbdata)
        artistdata['thumb'] = thumbs
    return artistdata

def allmusic_artistalbums(data):
    if False:
        return 10
    data = data.decode('utf-8')
    albums = []
    albumdata = re.search('tbody>\\s*(.*?)\\s*</tbody', data, re.S)
    if albumdata:
        albumlist = re.findall('tr.*?>\\s*(.*?)\\s*</tr', albumdata.group(1), re.S)
        if albumlist:
            for album in albumlist:
                albumdata = {}
                title = re.search('<a.*?>(.*?)<', album)
                if title:
                    albumdata['title'] = title.group(1)
                year = re.search('class="year".*?>\\s*(.*?)\\s*<', album)
                if year:
                    albumdata['year'] = year.group(1)
                else:
                    albumdata['year'] = ''
                if albumdata:
                    albums.append(albumdata)
    return albums