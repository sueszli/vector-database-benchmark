import asyncio
import logging
import platform
import sys
import lbry.wallet
from lbry.conf import TranscodeConfig
from lbry.file_analysis import VideoFileAnalyzer

def enable_logging():
    if False:
        for i in range(10):
            print('nop')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

async def process_video(analyzer, video_file):
    try:
        await analyzer.verify_or_repair(True, False, video_file)
        print('No concerns. Ship it!')
    except (FileNotFoundError, ValueError) as e:
        print('Analysis failed.', str(e))
    except Exception as e:
        print(str(e))
        transcode = input('Would you like to make a repaired clone now? [y/N] ')
        if transcode == 'y':
            try:
                (new_video_file, _) = await analyzer.verify_or_repair(True, True, video_file)
                print('Successfully created ', new_video_file)
            except Exception as e:
                print('Unable to complete the transcode. Message: ', str(e))

def main():
    if False:
        i = 10
        return i + 15
    if len(sys.argv) < 2:
        print('Usage: check_video.py <path to video file>', file=sys.stderr)
        sys.exit(1)
    enable_logging()
    video_file = sys.argv[1]
    conf = TranscodeConfig()
    analyzer = VideoFileAnalyzer(conf)
    try:
        asyncio.run(process_video(analyzer, video_file))
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()