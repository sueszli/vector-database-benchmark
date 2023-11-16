from .common import InfoExtractor
from ..utils import ExtractorError, classproperty, remove_start

class UnsupportedInfoExtractor(InfoExtractor):
    IE_DESC = False
    URLS = ()

    @classproperty
    def IE_NAME(cls):
        if False:
            for i in range(10):
                print('nop')
        return remove_start(super().IE_NAME, 'Known')

    @classproperty
    def _VALID_URL(cls):
        if False:
            i = 10
            return i + 15
        return f"https?://(?:www\\.)?(?:{'|'.join(cls.URLS)})"
LF = '\n       '

class KnownDRMIE(UnsupportedInfoExtractor):
    """Sites that are known to use DRM for all their videos

    Add to this list only if:
    * You are reasonably certain that the site uses DRM for ALL their videos
    * Multiple users have asked about this site on github/reddit/discord
    """
    URLS = ('play\\.hbomax\\.com', 'channel(?:4|5)\\.com', 'peacocktv\\.com', '(?:[\\w\\.]+\\.)?disneyplus\\.com', 'open\\.spotify\\.com/(?:track|playlist|album|artist)', 'tvnz\\.co\\.nz', 'oneplus\\.ch', 'artstation\\.com/learning/courses', 'philo\\.com', '(?:[\\w\\.]+\\.)?mech-plus\\.com', 'aha\\.video', 'mubi\\.com', 'vootkids\\.com', 'nowtv\\.it/watch', 'tv\\.apple\\.com', 'primevideo\\.com', 'hulu\\.com', 'resource\\.inkryptvideos\\.com', 'joyn\\.de', 'amazon\\.(?:\\w{2}\\.)?\\w+/gp/video', 'music\\.amazon\\.(?:\\w{2}\\.)?\\w+', '(?:watch|front)\\.njpwworld\\.com')
    _TESTS = [{'url': 'https://peacocktv.com/watch/playback/vod/GMO_00000000073159_01/f9d03003-eb04-3c7f-a7b6-a83ab7eb55bc', 'only_matching': True}, {'url': 'https://www.channel4.com/programmes/gurren-lagann/on-demand/69960-001', 'only_matching': True}, {'url': 'https://www.channel5.com/show/uk-s-strongest-man-2021/season-2021/episode-1', 'only_matching': True}, {'url': 'https://hsesn.apps.disneyplus.com', 'only_matching': True}, {'url': 'https://www.disneyplus.com', 'only_matching': True}, {'url': 'https://open.spotify.com/artist/', 'only_matching': True}, {'url': 'https://open.spotify.com/track/', 'only_matching': True}, {'url': 'https://www.tvnz.co.nz/shows/ice-airport-alaska/episodes/s1-e1', 'only_matching': True}, {'url': 'https://www.oneplus.ch/play/1008188', 'only_matching': True}, {'url': 'https://www.artstation.com/learning/courses/dqQ/character-design-masterclass-with-serge-birault/chapters/Rxn3/introduction', 'only_matching': True}, {'url': 'https://www.philo.com/player/player/vod/Vk9EOjYwODU0ODg5OTY0ODY0OTQ5NA', 'only_matching': True}, {'url': 'https://www.mech-plus.com/player/24892/stream?assetType=episodes&playlist_id=6', 'only_matching': True}, {'url': 'https://watch.mech-plus.com/details/25240?playlist_id=6', 'only_matching': True}, {'url': 'https://www.aha.video/player/movie/lucky-man', 'only_matching': True}, {'url': 'https://mubi.com/films/the-night-doctor', 'only_matching': True}, {'url': 'https://www.vootkids.com/movies/chhota-bheem-the-rise-of-kirmada/764459', 'only_matching': True}, {'url': 'https://www.nowtv.it/watch/home/asset/and-just-like-that/skyserie_f8fe979772e8437d8a61ab83b6d293e9/seasons/1/episodes/8/R_126182_HD', 'only_matching': True}, {'url': 'https://tv.apple.com/it/show/loot---una-fortuna/umc.cmc.5erbujil1mpazuerhr1udnk45?ctx_brand=tvs.sbd.4000', 'only_matching': True}, {'url': 'https://www.joyn.de/play/serien/clannad/1-1-wo-die-kirschblueten-fallen', 'only_matching': True}, {'url': 'https://music.amazon.co.jp/albums/B088Y368TK', 'only_matching': True}, {'url': 'https://www.amazon.co.jp/gp/video/detail/B09X5HBYRS/', 'only_matching': True}, {'url': 'https://www.primevideo.com/region/eu/detail/0H3DDB4KBJFNDCKKLHNRLRLVKQ/ref=atv_br_def_r_br_c_unkc_1_10', 'only_matching': True}, {'url': 'https://resource.inkryptvideos.com/v2-a83ns52/iframe/index.html#video_id=7999ea0f6e03439eb40d056258c2d736&otp=xxx', 'only_matching': True}, {'url': 'https://www.hulu.com/movie/anthem-6b25fac9-da2b-45a3-8e09-e4156b0471cc', 'only_matching': True}, {'url': 'https://watch.njpwworld.com/player/36447/series?assetType=series', 'only_matching': True}, {'url': 'https://front.njpwworld.com/p/s_series_00563_16_bs', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        raise ExtractorError(f"The requested site is known to use DRM protection. It will {self._downloader._format_err('NOT', self._downloader.Styles.EMPHASIS)} be supported.{LF}Please {self._downloader._format_err('DO NOT', self._downloader.Styles.ERROR)} open an issue, unless you have evidence that the video is not DRM protected", expected=True)

class KnownPiracyIE(UnsupportedInfoExtractor):
    """Sites that have been deemed to be piracy

    In order for this to not end up being a catalog of piracy sites,
    only sites that were once supported should be added to this list
    """
    URLS = ('dood\\.(?:to|watch|so|pm|wf|re)', 'viewsb\\.com', 'filemoon\\.sx', 'hentai\\.animestigma\\.com', 'thisav\\.com')
    _TESTS = [{'url': 'http://dood.to/e/5s1wmbdacezb', 'only_matching': True}, {'url': 'https://thisav.com/en/terms', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            print('Hello World!')
        raise ExtractorError(f"This website is no longer supported since it has been determined to be primarily used for piracy.{LF}{self._downloader._format_err('DO NOT', self._downloader.Styles.ERROR)} open issues for it", expected=True)