import concurrent.futures
import hashlib
import logging
from embedchain.loaders.base_loader import BaseLoader
from embedchain.loaders.youtube_video import YoutubeVideoLoader

class YoutubeChannelLoader(BaseLoader):
    """Loader for youtube channel."""

    def load_data(self, channel_name):
        if False:
            print('Hello World!')
        try:
            import yt_dlp
        except ImportError as e:
            raise ValueError("YoutubeLoader requires extra dependencies. Install with `pip install --upgrade 'embedchain[youtube_channel]'`") from e
        data = []
        data_urls = []
        youtube_url = f'https://www.youtube.com/{channel_name}/videos'
        youtube_video_loader = YoutubeVideoLoader()

        def _get_yt_video_links():
            if False:
                print('Hello World!')
            try:
                ydl_opts = {'quiet': True, 'extract_flat': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(youtube_url, download=False)
                    if 'entries' in info_dict:
                        videos = [entry['url'] for entry in info_dict['entries']]
                        return videos
            except Exception:
                logging.error(f'Failed to fetch youtube videos for channel: {channel_name}')
                return []

        def _load_yt_video(video_link):
            if False:
                for i in range(10):
                    print('nop')
            try:
                each_load_data = youtube_video_loader.load_data(video_link)
                if each_load_data:
                    return each_load_data.get('data')
            except Exception as e:
                logging.error(f'Failed to load youtube video {video_link}: {e}')
            return None

        def _add_youtube_channel():
            if False:
                i = 10
                return i + 15
            video_links = _get_yt_video_links()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_video = {executor.submit(_load_yt_video, video_link): video_link for video_link in video_links}
                for future in concurrent.futures.as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        results = future.result()
                        if results:
                            data.extend(results)
                            data_urls.extend([result.get('meta_data').get('url') for result in results])
                    except Exception as e:
                        logging.error(f'Failed to process youtube video {video}: {e}')
        _add_youtube_channel()
        doc_id = hashlib.sha256((youtube_url + ', '.join(data_urls)).encode()).hexdigest()
        return {'doc_id': doc_id, 'data': data}