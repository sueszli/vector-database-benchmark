import re
from .common import InfoExtractor
from ..compat import compat_str
from ..utils import ExtractorError, determine_ext, int_or_none, float_or_none, js_to_json, orderedSet, strip_jsonp, strip_or_none, traverse_obj, unified_strdate, url_or_none, US_RATINGS

class PBSIE(InfoExtractor):
    _STATIONS = (('(?:video|www|player)\\.pbs\\.org', 'PBS: Public Broadcasting Service'), ('video\\.aptv\\.org', 'APT - Alabama Public Television (WBIQ)'), ('video\\.gpb\\.org', 'GPB/Georgia Public Broadcasting (WGTV)'), ('video\\.mpbonline\\.org', 'Mississippi Public Broadcasting (WMPN)'), ('video\\.wnpt\\.org', 'Nashville Public Television (WNPT)'), ('video\\.wfsu\\.org', 'WFSU-TV (WFSU)'), ('video\\.wsre\\.org', 'WSRE (WSRE)'), ('video\\.wtcitv\\.org', 'WTCI (WTCI)'), ('video\\.pba\\.org', 'WPBA/Channel 30 (WPBA)'), ('video\\.alaskapublic\\.org', 'Alaska Public Media (KAKM)'), ('video\\.azpbs\\.org', 'Arizona PBS (KAET)'), ('portal\\.knme\\.org', 'KNME-TV/Channel 5 (KNME)'), ('video\\.vegaspbs\\.org', 'Vegas PBS (KLVX)'), ('watch\\.aetn\\.org', 'AETN/ARKANSAS ETV NETWORK (KETS)'), ('video\\.ket\\.org', 'KET (WKLE)'), ('video\\.wkno\\.org', 'WKNO/Channel 10 (WKNO)'), ('video\\.lpb\\.org', 'LPB/LOUISIANA PUBLIC BROADCASTING (WLPB)'), ('videos\\.oeta\\.tv', 'OETA (KETA)'), ('video\\.optv\\.org', 'Ozarks Public Television (KOZK)'), ('watch\\.wsiu\\.org', 'WSIU Public Broadcasting (WSIU)'), ('video\\.keet\\.org', 'KEET TV (KEET)'), ('pbs\\.kixe\\.org', 'KIXE/Channel 9 (KIXE)'), ('video\\.kpbs\\.org', 'KPBS San Diego (KPBS)'), ('video\\.kqed\\.org', 'KQED (KQED)'), ('vids\\.kvie\\.org', 'KVIE Public Television (KVIE)'), ('video\\.pbssocal\\.org', 'PBS SoCal/KOCE (KOCE)'), ('video\\.valleypbs\\.org', 'ValleyPBS (KVPT)'), ('video\\.cptv\\.org', 'CONNECTICUT PUBLIC TELEVISION (WEDH)'), ('watch\\.knpb\\.org', 'KNPB Channel 5 (KNPB)'), ('video\\.soptv\\.org', 'SOPTV (KSYS)'), ('video\\.rmpbs\\.org', 'Rocky Mountain PBS (KRMA)'), ('video\\.kenw\\.org', 'KENW-TV3 (KENW)'), ('video\\.kued\\.org', 'KUED Channel 7 (KUED)'), ('video\\.wyomingpbs\\.org', 'Wyoming PBS (KCWC)'), ('video\\.cpt12\\.org', 'Colorado Public Television / KBDI 12 (KBDI)'), ('video\\.kbyueleven\\.org', 'KBYU-TV (KBYU)'), ('video\\.thirteen\\.org', 'Thirteen/WNET New York (WNET)'), ('video\\.wgbh\\.org', 'WGBH/Channel 2 (WGBH)'), ('video\\.wgby\\.org', 'WGBY (WGBY)'), ('watch\\.njtvonline\\.org', 'NJTV Public Media NJ (WNJT)'), ('watch\\.wliw\\.org', 'WLIW21 (WLIW)'), ('video\\.mpt\\.tv', 'mpt/Maryland Public Television (WMPB)'), ('watch\\.weta\\.org', 'WETA Television and Radio (WETA)'), ('video\\.whyy\\.org', 'WHYY (WHYY)'), ('video\\.wlvt\\.org', 'PBS 39 (WLVT)'), ('video\\.wvpt\\.net', 'WVPT - Your Source for PBS and More! (WVPT)'), ('video\\.whut\\.org', 'Howard University Television (WHUT)'), ('video\\.wedu\\.org', 'WEDU PBS (WEDU)'), ('video\\.wgcu\\.org', 'WGCU Public Media (WGCU)'), ('video\\.wpbt2\\.org', 'WPBT2 (WPBT)'), ('video\\.wucftv\\.org', 'WUCF TV (WUCF)'), ('video\\.wuft\\.org', 'WUFT/Channel 5 (WUFT)'), ('watch\\.wxel\\.org', 'WXEL/Channel 42 (WXEL)'), ('video\\.wlrn\\.org', 'WLRN/Channel 17 (WLRN)'), ('video\\.wusf\\.usf\\.edu', 'WUSF Public Broadcasting (WUSF)'), ('video\\.scetv\\.org', 'ETV (WRLK)'), ('video\\.unctv\\.org', 'UNC-TV (WUNC)'), ('video\\.pbshawaii\\.org', 'PBS Hawaii - Oceanic Cable Channel 10 (KHET)'), ('video\\.idahoptv\\.org', 'Idaho Public Television (KAID)'), ('video\\.ksps\\.org', 'KSPS (KSPS)'), ('watch\\.opb\\.org', 'OPB (KOPB)'), ('watch\\.nwptv\\.org', 'KWSU/Channel 10 & KTNW/Channel 31 (KWSU)'), ('video\\.will\\.illinois\\.edu', 'WILL-TV (WILL)'), ('video\\.networkknowledge\\.tv', 'Network Knowledge - WSEC/Springfield (WSEC)'), ('video\\.wttw\\.com', 'WTTW11 (WTTW)'), ('video\\.iptv\\.org', 'Iowa Public Television/IPTV (KDIN)'), ('video\\.ninenet\\.org', 'Nine Network (KETC)'), ('video\\.wfwa\\.org', 'PBS39 Fort Wayne (WFWA)'), ('video\\.wfyi\\.org', 'WFYI Indianapolis (WFYI)'), ('video\\.mptv\\.org', 'Milwaukee Public Television (WMVS)'), ('video\\.wnin\\.org', 'WNIN (WNIN)'), ('video\\.wnit\\.org', 'WNIT Public Television (WNIT)'), ('video\\.wpt\\.org', 'WPT (WPNE)'), ('video\\.wvut\\.org', 'WVUT/Channel 22 (WVUT)'), ('video\\.weiu\\.net', 'WEIU/Channel 51 (WEIU)'), ('video\\.wqpt\\.org', 'WQPT-TV (WQPT)'), ('video\\.wycc\\.org', 'WYCC PBS Chicago (WYCC)'), ('video\\.wipb\\.org', 'WIPB-TV (WIPB)'), ('video\\.indianapublicmedia\\.org', 'WTIU (WTIU)'), ('watch\\.cetconnect\\.org', 'CET  (WCET)'), ('video\\.thinktv\\.org', 'ThinkTVNetwork (WPTD)'), ('video\\.wbgu\\.org', 'WBGU-TV (WBGU)'), ('video\\.wgvu\\.org', 'WGVU TV (WGVU)'), ('video\\.netnebraska\\.org', 'NET1 (KUON)'), ('video\\.pioneer\\.org', 'Pioneer Public Television (KWCM)'), ('watch\\.sdpb\\.org', 'SDPB Television (KUSD)'), ('video\\.tpt\\.org', 'TPT (KTCA)'), ('watch\\.ksmq\\.org', 'KSMQ (KSMQ)'), ('watch\\.kpts\\.org', 'KPTS/Channel 8 (KPTS)'), ('watch\\.ktwu\\.org', 'KTWU/Channel 11 (KTWU)'), ('watch\\.easttennesseepbs\\.org', 'East Tennessee PBS (WSJK)'), ('video\\.wcte\\.tv', 'WCTE-TV (WCTE)'), ('video\\.wljt\\.org', 'WLJT, Channel 11 (WLJT)'), ('video\\.wosu\\.org', 'WOSU TV (WOSU)'), ('video\\.woub\\.org', 'WOUB/WOUC (WOUB)'), ('video\\.wvpublic\\.org', 'WVPB (WVPB)'), ('video\\.wkyupbs\\.org', 'WKYU-PBS (WKYU)'), ('video\\.kera\\.org', 'KERA 13 (KERA)'), ('video\\.mpbn\\.net', 'MPBN (WCBB)'), ('video\\.mountainlake\\.org', 'Mountain Lake PBS (WCFE)'), ('video\\.nhptv\\.org', 'NHPTV (WENH)'), ('video\\.vpt\\.org', 'Vermont PBS (WETK)'), ('video\\.witf\\.org', 'witf (WITF)'), ('watch\\.wqed\\.org', 'WQED Multimedia (WQED)'), ('video\\.wmht\\.org', 'WMHT Educational Telecommunications (WMHT)'), ('video\\.deltabroadcasting\\.org', 'Q-TV (WDCQ)'), ('video\\.dptv\\.org', 'WTVS Detroit Public TV (WTVS)'), ('video\\.wcmu\\.org', 'CMU Public Television (WCMU)'), ('video\\.wkar\\.org', 'WKAR-TV (WKAR)'), ('wnmuvideo\\.nmu\\.edu', 'WNMU-TV Public TV 13 (WNMU)'), ('video\\.wdse\\.org', 'WDSE - WRPT (WDSE)'), ('video\\.wgte\\.org', 'WGTE TV (WGTE)'), ('video\\.lptv\\.org', 'Lakeland Public Television (KAWE)'), ('video\\.kmos\\.org', 'KMOS-TV - Channels 6.1, 6.2 and 6.3 (KMOS)'), ('watch\\.montanapbs\\.org', 'MontanaPBS (KUSM)'), ('video\\.krwg\\.org', 'KRWG/Channel 22 (KRWG)'), ('video\\.kacvtv\\.org', 'KACV (KACV)'), ('video\\.kcostv\\.org', 'KCOS/Channel 13 (KCOS)'), ('video\\.wcny\\.org', 'WCNY/Channel 24 (WCNY)'), ('video\\.wned\\.org', 'WNED (WNED)'), ('watch\\.wpbstv\\.org', 'WPBS (WPBS)'), ('video\\.wskg\\.org', 'WSKG Public TV (WSKG)'), ('video\\.wxxi\\.org', 'WXXI (WXXI)'), ('video\\.wpsu\\.org', 'WPSU (WPSU)'), ('on-demand\\.wvia\\.org', 'WVIA Public Media Studios (WVIA)'), ('video\\.wtvi\\.org', 'WTVI (WTVI)'), ('video\\.westernreservepublicmedia\\.org', 'Western Reserve PBS (WNEO)'), ('video\\.ideastream\\.org', 'WVIZ/PBS ideastream (WVIZ)'), ('video\\.kcts9\\.org', 'KCTS 9 (KCTS)'), ('video\\.basinpbs\\.org', 'Basin PBS (KPBT)'), ('video\\.houstonpbs\\.org', 'KUHT / Channel 8 (KUHT)'), ('video\\.klrn\\.org', 'KLRN (KLRN)'), ('video\\.klru\\.tv', 'KLRU (KLRU)'), ('video\\.wtjx\\.org', 'WTJX Channel 12 (WTJX)'), ('video\\.ideastations\\.org', 'WCVE PBS (WCVE)'), ('video\\.kbtc\\.org', 'KBTC Public Television (KBTC)'))
    IE_NAME = 'pbs'
    IE_DESC = 'Public Broadcasting Service (PBS) and member stations: %s' % ', '.join(list(zip(*_STATIONS))[1])
    _VALID_URL = '(?x)https?://\n        (?:\n           # Direct video URL\n           (?:%s)/(?:(?:vir|port)alplayer|video)/(?P<id>[0-9]+)(?:[?/]|$) |\n           # Article with embedded player (or direct video)\n           (?:www\\.)?pbs\\.org/(?:[^/]+/){1,5}(?P<presumptive_id>[^/]+?)(?:\\.html)?/?(?:$|[?\\#]) |\n           # Player\n           (?:video|player)\\.pbs\\.org/(?:widget/)?partnerplayer/(?P<player_id>[^/]+)\n        )\n    ' % '|'.join(list(zip(*_STATIONS))[0])
    _GEO_COUNTRIES = ['US']
    _TESTS = [{'url': 'http://www.pbs.org/tpt/constitution-usa-peter-sagal/watch/a-more-perfect-union/', 'md5': '173dc391afd361fa72eab5d3d918968d', 'info_dict': {'id': '2365006249', 'ext': 'mp4', 'title': 'Constitution USA with Peter Sagal - A More Perfect Union', 'description': 'md5:31b664af3c65fd07fa460d306b837d00', 'duration': 3190}}, {'url': 'http://www.pbs.org/wgbh/pages/frontline/losing-iraq/', 'md5': '6f722cb3c3982186d34b0f13374499c7', 'info_dict': {'id': '2365297690', 'ext': 'mp4', 'title': 'FRONTLINE - Losing Iraq', 'description': 'md5:5979a4d069b157f622d02bff62fbe654', 'duration': 5050}}, {'url': 'http://www.pbs.org/newshour/bb/education-jan-june12-cyberschools_02-23/', 'md5': 'b19856d7f5351b17a5ab1dc6a64be633', 'info_dict': {'id': '2201174722', 'ext': 'mp4', 'title': 'PBS NewsHour - Cyber Schools Gain Popularity, but Quality Questions Persist', 'description': 'md5:86ab9a3d04458b876147b355788b8781', 'duration': 801}}, {'url': 'http://www.pbs.org/wnet/gperf/dudamel-conducts-verdi-requiem-hollywood-bowl-full-episode/3374/', 'md5': 'c62859342be2a0358d6c9eb306595978', 'info_dict': {'id': '2365297708', 'ext': 'mp4', 'title': 'Great Performances - Dudamel Conducts Verdi Requiem at the Hollywood Bowl - Full', 'description': 'md5:657897370e09e2bc6bf0f8d2cd313c6b', 'duration': 6559, 'thumbnail': 're:^https?://.*\\.jpg$'}}, {'url': 'http://www.pbs.org/wgbh/nova/earth/killer-typhoon.html', 'md5': '908f3e5473a693b266b84e25e1cf9703', 'info_dict': {'id': '2365160389', 'display_id': 'killer-typhoon', 'ext': 'mp4', 'description': 'md5:c741d14e979fc53228c575894094f157', 'title': 'NOVA - Killer Typhoon', 'duration': 3172, 'thumbnail': 're:^https?://.*\\.jpg$', 'upload_date': '20140122', 'age_limit': 10}}, {'url': 'http://www.pbs.org/wgbh/pages/frontline/united-states-of-secrets/', 'info_dict': {'id': 'united-states-of-secrets'}, 'playlist_count': 2}, {'url': 'http://www.pbs.org/wgbh/americanexperience/films/great-war/', 'info_dict': {'id': 'great-war'}, 'playlist_count': 3}, {'url': 'http://www.pbs.org/wgbh/americanexperience/films/death/player/', 'info_dict': {'id': '2276541483', 'display_id': 'player', 'ext': 'mp4', 'title': 'American Experience - Death and the Civil War, Chapter 1', 'description': 'md5:67fa89a9402e2ee7d08f53b920674c18', 'duration': 682, 'thumbnail': 're:^https?://.*\\.jpg$'}, 'params': {'skip_download': True}}, {'url': 'http://www.pbs.org/video/2365245528/', 'md5': '115223d41bd55cda8ae5cd5ed4e11497', 'info_dict': {'id': '2365245528', 'display_id': '2365245528', 'ext': 'mp4', 'title': 'FRONTLINE - United States of Secrets (Part One)', 'description': 'md5:55756bd5c551519cc4b7703e373e217e', 'duration': 6851, 'thumbnail': 're:^https?://.*\\.jpg$'}}, {'url': 'http://www.pbs.org/food/features/a-chefs-life-season-3-episode-5-prickly-business/', 'md5': '59b0ef5009f9ac8a319cc5efebcd865e', 'info_dict': {'id': '2365546844', 'display_id': 'a-chefs-life-season-3-episode-5-prickly-business', 'ext': 'mp4', 'title': "A Chef's Life - Season 3, Ep. 5: Prickly Business", 'description': 'md5:c0ff7475a4b70261c7e58f493c2792a5', 'duration': 1480, 'thumbnail': 're:^https?://.*\\.jpg$'}}, {'url': 'http://www.pbs.org/wgbh/pages/frontline/the-atomic-artists', 'info_dict': {'id': '2070868960', 'display_id': 'the-atomic-artists', 'ext': 'mp4', 'title': 'FRONTLINE - The Atomic Artists', 'description': 'md5:f677e4520cfacb4a5ce1471e31b57800', 'duration': 723, 'thumbnail': 're:^https?://.*\\.jpg$'}, 'params': {'skip_download': True}}, {'url': 'http://www.pbs.org/video/2365641075/', 'md5': 'fdf907851eab57211dd589cf12006666', 'info_dict': {'id': '2365641075', 'ext': 'mp4', 'title': 'FRONTLINE - Netanyahu at War', 'duration': 6852, 'thumbnail': 're:^https?://.*\\.jpg$', 'formats': 'mincount:8'}}, {'url': 'https://www.pbs.org/video/pbs-newshour-full-episode-july-31-2017-1501539057/', 'info_dict': {'id': '3003333873', 'ext': 'mp4', 'title': 'PBS NewsHour - full episode July 31, 2017', 'description': 'md5:d41d8cd98f00b204e9800998ecf8427e', 'duration': 3265, 'thumbnail': 're:^https?://.*\\.jpg$'}, 'params': {'skip_download': True}}, {'url': 'http://www.pbs.org/wgbh/roadshow/watch/episode/2105-indianapolis-hour-2/', 'info_dict': {'id': '2365936247', 'ext': 'mp4', 'title': 'Antiques Roadshow - Indianapolis, Hour 2', 'description': 'md5:524b32249db55663e7231b6b8d1671a2', 'duration': 3180, 'thumbnail': 're:^https?://.*\\.jpg$'}, 'params': {'skip_download': True}, 'expected_warnings': ['HTTP Error 403: Forbidden']}, {'url': 'https://www.pbs.org/wgbh/masterpiece/episodes/victoria-s2-e1/', 'info_dict': {'id': '3007193718', 'ext': 'mp4', 'title': "Victoria - A Soldier's Daughter / The Green-Eyed Monster", 'description': 'md5:37efbac85e0c09b009586523ec143652', 'duration': 6292, 'thumbnail': 're:^https?://.*\\.(?:jpg|JPG)$'}, 'params': {'skip_download': True}, 'expected_warnings': ['HTTP Error 403: Forbidden']}, {'url': 'https://player.pbs.org/partnerplayer/tOz9tM5ljOXQqIIWke53UA==/', 'info_dict': {'id': '3011407934', 'ext': 'mp4', 'title': 'Stories from the Stage - Road Trip', 'duration': 1619, 'thumbnail': 're:^https?://.*\\.(?:jpg|JPG)$'}, 'params': {'skip_download': True}, 'expected_warnings': ['HTTP Error 403: Forbidden']}, {'url': 'http://player.pbs.org/widget/partnerplayer/2365297708/?start=0&end=0&chapterbar=false&endscreen=false&topbar=true', 'only_matching': True}, {'url': 'http://watch.knpb.org/video/2365616055/', 'only_matching': True}, {'url': 'https://player.pbs.org/portalplayer/3004638221/?uid=', 'only_matching': True}]
    _ERRORS = {101: "We're sorry, but this video is not yet available.", 403: "We're sorry, but this video is not available in your region due to right restrictions.", 404: 'We are experiencing technical difficulties that are preventing us from playing the video at this time. Please check back again soon.', 410: 'This video has expired and is no longer available for online streaming.'}

    def _real_initialize(self):
        if False:
            for i in range(10):
                print('nop')
        cookie = (self._download_json('http://localization.services.pbs.org/localize/auto/cookie/', None, headers=self.geo_verification_headers(), fatal=False) or {}).get('cookie')
        if cookie:
            station = self._search_regex('#?s=\\["([^"]+)"', cookie, 'station')
            if station:
                self._set_cookie('.pbs.org', 'pbsol.station', station)

    def _extract_webpage(self, url):
        if False:
            while True:
                i = 10
        mobj = self._match_valid_url(url)
        description = None
        presumptive_id = mobj.group('presumptive_id')
        display_id = presumptive_id
        if presumptive_id:
            webpage = self._download_webpage(url, display_id)
            description = strip_or_none(self._og_search_description(webpage, default=None) or self._html_search_meta('description', webpage, default=None))
            upload_date = unified_strdate(self._search_regex('<input type="hidden" id="air_date_[0-9]+" value="([^"]+)"', webpage, 'upload date', default=None))
            MULTI_PART_REGEXES = ('<div[^>]+class="videotab[^"]*"[^>]+vid="(\\d+)"', '<a[^>]+href=["\\\']#(?:video-|part)\\d+["\\\'][^>]+data-cove[Ii]d=["\\\'](\\d+)')
            for p in MULTI_PART_REGEXES:
                tabbed_videos = orderedSet(re.findall(p, webpage))
                if tabbed_videos:
                    return (tabbed_videos, presumptive_id, upload_date, description)
            MEDIA_ID_REGEXES = ["div\\s*:\\s*'videoembed'\\s*,\\s*mediaid\\s*:\\s*'(\\d+)'", 'class="coveplayerid">([^<]+)<', '<section[^>]+data-coveid="(\\d+)"', '<input type="hidden" id="pbs_video_id_[0-9]+" value="([0-9]+)"/>', "(?s)window\\.PBS\\.playerConfig\\s*=\\s*{.*?id\\s*:\\s*'([0-9]+)',", '<div[^>]+\\bdata-cove-id=["\\\'](\\d+)"', '<iframe[^>]+\\bsrc=["\\\'](?:https?:)?//video\\.pbs\\.org/widget/partnerplayer/(\\d+)']
            media_id = self._search_regex(MEDIA_ID_REGEXES, webpage, 'media ID', fatal=False, default=None)
            if media_id:
                return (media_id, presumptive_id, upload_date, description)
            video_id = self._search_regex('videoid\\s*:\\s*"([\\d+a-z]{7,})"', webpage, 'videoid', default=None)
            if video_id:
                prg_id = self._search_regex('videoid\\s*:\\s*"([\\d+a-z]{7,})"', webpage, 'videoid')[7:]
                if 'q' in prg_id:
                    prg_id = prg_id.split('q')[1]
                prg_id = int(prg_id, 16)
                getdir = self._download_json('http://www.pbs.org/wgbh/pages/frontline/.json/getdir/getdir%d.json' % prg_id, presumptive_id, 'Downloading getdir JSON', transform_source=strip_jsonp)
                return (getdir['mid'], presumptive_id, upload_date, description)
            for iframe in re.findall('(?s)<iframe(.+?)></iframe>', webpage):
                url = self._search_regex('src=(["\\\'])(?P<url>.+?partnerplayer.+?)\\1', iframe, 'player URL', default=None, group='url')
                if url:
                    break
            if not url:
                url = self._og_search_url(webpage)
            mobj = re.match(self._VALID_URL, self._proto_relative_url(url.strip()))
        player_id = mobj.group('player_id')
        if not display_id:
            display_id = player_id
        if player_id:
            player_page = self._download_webpage(url, display_id, note='Downloading player page', errnote='Could not download player page')
            video_id = self._search_regex('<div\\s+id=["\\\']video_(\\d+)', player_page, 'video ID', default=None)
            if not video_id:
                video_info = self._extract_video_data(player_page, 'video data', display_id)
                video_id = compat_str(video_info.get('id') or video_info['contentID'])
        else:
            video_id = mobj.group('id')
            display_id = video_id
        return (video_id, display_id, None, description)

    def _extract_video_data(self, string, name, video_id, fatal=True):
        if False:
            for i in range(10):
                print('nop')
        return self._parse_json(self._search_regex(['(?s)PBS\\.videoData\\s*=\\s*({.+?});\\n', 'window\\.videoBridge\\s*=\\s*({.+?});'], string, name, default='{}'), video_id, transform_source=js_to_json, fatal=fatal)

    def _real_extract(self, url):
        if False:
            return 10
        (video_id, display_id, upload_date, description) = self._extract_webpage(url)
        if isinstance(video_id, list):
            entries = [self.url_result('http://video.pbs.org/video/%s' % vid_id, 'PBS', vid_id) for vid_id in video_id]
            return self.playlist_result(entries, display_id)
        info = {}
        redirects = []
        redirect_urls = set()

        def extract_redirect_urls(info):
            if False:
                for i in range(10):
                    print('nop')
            for encoding_name in ('recommended_encoding', 'alternate_encoding'):
                redirect = info.get(encoding_name)
                if not redirect:
                    continue
                redirect_url = redirect.get('url')
                if redirect_url and redirect_url not in redirect_urls:
                    redirects.append(redirect)
                    redirect_urls.add(redirect_url)
            encodings = info.get('encodings')
            if isinstance(encodings, list):
                for encoding in encodings:
                    encoding_url = url_or_none(encoding)
                    if encoding_url and encoding_url not in redirect_urls:
                        redirects.append({'url': encoding_url})
                        redirect_urls.add(encoding_url)
        chapters = []
        for page in ('widget/partnerplayer', 'portalplayer'):
            player = self._download_webpage('http://player.pbs.org/%s/%s' % (page, video_id), display_id, 'Downloading %s page' % page, fatal=False)
            if player:
                video_info = self._extract_video_data(player, '%s video data' % page, display_id, fatal=False)
                if video_info:
                    extract_redirect_urls(video_info)
                    if not info:
                        info = video_info
                if not chapters:
                    raw_chapters = video_info.get('chapters') or []
                    if not raw_chapters:
                        for chapter_data in re.findall('(?s)chapters\\.push\\(({.*?})\\)', player):
                            chapter = self._parse_json(chapter_data, video_id, js_to_json, fatal=False)
                            if not chapter:
                                continue
                            raw_chapters.append(chapter)
                    for chapter in raw_chapters:
                        start_time = float_or_none(chapter.get('start_time'), 1000)
                        duration = float_or_none(chapter.get('duration'), 1000)
                        if start_time is None or duration is None:
                            continue
                        chapters.append({'start_time': start_time, 'end_time': start_time + duration, 'title': chapter.get('title')})
        formats = []
        http_url = None
        hls_subs = {}
        for (num, redirect) in enumerate(redirects):
            redirect_id = redirect.get('eeid')
            redirect_info = self._download_json('%s?format=json' % redirect['url'], display_id, 'Downloading %s video url info' % (redirect_id or num), headers=self.geo_verification_headers())
            if redirect_info['status'] == 'error':
                message = self._ERRORS.get(redirect_info['http_code'], redirect_info['message'])
                if redirect_info['http_code'] == 403:
                    self.raise_geo_restricted(msg=message, countries=self._GEO_COUNTRIES)
                raise ExtractorError('%s said: %s' % (self.IE_NAME, message), expected=True)
            format_url = redirect_info.get('url')
            if not format_url:
                continue
            if determine_ext(format_url) == 'm3u8':
                (hls_formats, hls_subs) = self._extract_m3u8_formats_and_subtitles(format_url, display_id, 'mp4', m3u8_id='hls', fatal=False)
                formats.extend(hls_formats)
            else:
                formats.append({'url': format_url, 'format_id': redirect_id})
                if re.search('^https?://.*(?:\\d+k|baseline)', format_url):
                    http_url = format_url
        self._remove_duplicate_formats(formats)
        m3u8_formats = list(filter(lambda f: f.get('protocol') == 'm3u8' and f.get('vcodec') != 'none', formats))
        if http_url:
            for m3u8_format in m3u8_formats:
                bitrate = self._search_regex('(\\d+)k', m3u8_format['url'], 'bitrate', default=None)
                if not bitrate or int(bitrate) < 400:
                    continue
                f_url = re.sub('\\d+k|baseline', bitrate + 'k', http_url)
                if not self._is_valid_url(f_url, display_id, 'http-%sk video' % bitrate):
                    continue
                f = m3u8_format.copy()
                f.update({'url': f_url, 'format_id': m3u8_format['format_id'].replace('hls', 'http'), 'protocol': 'http'})
                formats.append(f)
        for f in formats:
            if (f.get('format_note') or '').endswith(' AD'):
                f['language_preference'] = -10
        rating_str = info.get('rating')
        if rating_str is not None:
            rating_str = rating_str.rpartition('-')[2]
        age_limit = US_RATINGS.get(rating_str)
        subtitles = {}
        captions = info.get('cc') or {}
        for caption_url in captions.values():
            subtitles.setdefault('en', []).append({'url': caption_url})
        subtitles = self._merge_subtitles(subtitles, hls_subs)
        alt_title = info.get('program', {}).get('title')
        if alt_title:
            info['title'] = alt_title + ' - ' + re.sub('^' + alt_title + '[\\s\\-:]+', '', info['title'])
        description = info.get('description') or info.get('program', {}).get('description') or description
        return {'id': video_id, 'display_id': display_id, 'title': info['title'], 'description': description, 'thumbnail': info.get('image_url'), 'duration': int_or_none(info.get('duration')), 'age_limit': age_limit, 'upload_date': upload_date, 'formats': formats, 'subtitles': subtitles, 'chapters': chapters}

class PBSKidsIE(InfoExtractor):
    _VALID_URL = 'https?://(?:www\\.)?pbskids\\.org/video/[\\w-]+/(?P<id>\\d+)'
    _TESTS = [{'url': 'https://pbskids.org/video/molly-of-denali/3030407927', 'md5': '1ded20a017cc6b53446238f1804ce4c7', 'info_dict': {'id': '3030407927', 'title': 'Bird in the Hand/Bye-Bye Birdie', 'channel': 'molly-of-denali', 'duration': 1540, 'ext': 'mp4', 'series': 'Molly of Denali', 'description': 'md5:d006b2211633685d8ebc8d03b6d5611e', 'categories': ['Episode'], 'upload_date': '20190718'}}, {'url': 'https://pbskids.org/video/plum-landing/2365205059', 'md5': '92e5d189851a64ae1d0237a965be71f5', 'info_dict': {'id': '2365205059', 'title': "Cooper's Favorite Place in Nature", 'channel': 'plum-landing', 'duration': 67, 'ext': 'mp4', 'series': 'Plum Landing', 'description': 'md5:657e5fc4356a84ead1c061eb280ff05d', 'categories': ['Episode'], 'upload_date': '20140302'}}]

    def _real_extract(self, url):
        if False:
            i = 10
            return i + 15
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id)
        meta = self._search_json('window\\._PBS_KIDS_DEEPLINK\\s*=', webpage, 'video info', video_id)
        (formats, subtitles) = self._extract_m3u8_formats_and_subtitles(traverse_obj(meta, ('video_obj', 'URI', {url_or_none})), video_id, ext='mp4')
        return {'id': video_id, 'formats': formats, 'subtitles': subtitles, **traverse_obj(meta, {'categories': ('video_obj', 'video_type', {str}, {lambda x: [x] if x else None}), 'channel': ('show_slug', {str}), 'description': ('video_obj', 'description', {str}), 'duration': ('video_obj', 'duration', {int_or_none}), 'series': ('video_obj', 'program_title', {str}), 'title': ('video_obj', 'title', {str}), 'upload_date': ('video_obj', 'air_date', {unified_strdate})})}