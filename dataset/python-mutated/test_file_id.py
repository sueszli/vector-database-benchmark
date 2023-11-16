import pytest
from pyrogram.file_id import FileId, FileUniqueId, FileType, FileUniqueType

def check(file_id: str, expected_file_type: FileType):
    if False:
        for i in range(10):
            print('nop')
    decoded = FileId.decode(file_id)
    assert decoded.file_type == expected_file_type
    assert decoded.encode() == file_id

def check_unique(file_unique_id: str, expected_file_unique_type: FileUniqueType):
    if False:
        print('Hello World!')
    decoded = FileUniqueId.decode(file_unique_id)
    assert decoded.file_unique_type == expected_file_unique_type
    assert decoded.encode() == file_unique_id

def test_audio():
    if False:
        i = 10
        return i + 15
    audio = 'CQACAgIAAx0CAAGgr9AAAgmQX7b4XPBstC1fFUuJBooHTHFd7HMAAgUAA4GkuUnVOGG5P196yR4E'
    audio_unique = 'AgADBQADgaS5SQ'
    audio_thumb = 'AAMCAgADHQIAAaCv0AACCZBftvhc8Gy0LV8VS4kGigdMcV3scwACBQADgaS5SdU4Ybk_X3rJIH3qihAAAwEAB20AA_OeAQABHgQ'
    audio_thumb_unique = 'AQADIH3qihAAA_OeAQAB'
    check(audio, FileType.AUDIO)
    check_unique(audio_unique, FileUniqueType.DOCUMENT)
    check(audio_thumb, FileType.THUMBNAIL)
    check_unique(audio_thumb_unique, FileUniqueType.PHOTO)

def test_video():
    if False:
        return 10
    video = 'BAACAgIAAx0CAAGgr9AAAgmRX7b4Xv9f-4BK5VR_5ppIOF6UIp0AAgYAA4GkuUmhnZz2xC37wR4E'
    video_unique = 'AgADBgADgaS5SQ'
    video_thumb = 'AAMCAgADHQIAAaCv0AACCZFftvhe_1_7gErlVH_mmkg4XpQinQACBgADgaS5SaGdnPbELfvBIH3qihAAAwEAB20AA_WeAQABHgQ'
    video_thumb_unique = 'AQADIH3qihAAA_WeAQAB'
    check(video, FileType.VIDEO)
    check_unique(video_unique, FileUniqueType.DOCUMENT)
    check(video_thumb, FileType.THUMBNAIL)
    check_unique(video_thumb_unique, FileUniqueType.PHOTO)

def test_document():
    if False:
        for i in range(10):
            print('nop')
    document = 'BQACAgIAAx0CAAGgr9AAAgmPX7b4UxbjNoFEO_L0I4s6wrXNJA8AAgQAA4GkuUm9FFvIaOhXWR4E'
    document_unique = 'AgADBAADgaS5SQ'
    document_thumb = 'AAMCAgADHQIAAaCv0AACCY9ftvhTFuM2gUQ78vQjizrCtc0kDwACBAADgaS5Sb0UW8ho6FdZIH3qihAAAwEAB3MAA_GeAQABHgQ'
    document_thumb_unique = 'AQADIH3qihAAA_GeAQAB'
    check(document, FileType.DOCUMENT)
    check_unique(document_unique, FileUniqueType.DOCUMENT)
    check(document_thumb, FileType.THUMBNAIL)
    check_unique(document_thumb_unique, FileUniqueType.PHOTO)

def test_animation():
    if False:
        for i in range(10):
            print('nop')
    animation = 'CgACAgIAAx0CAAGgr9AAAgmSX7b4Y2g8_QW2XFd49iUmRnHOyG8AAgcAA4GkuUnry9gWDzF_5R4E'
    animation_unique = 'AgADBwADgaS5SQ'
    check(animation, FileType.ANIMATION)
    check_unique(animation_unique, FileUniqueType.DOCUMENT)

def test_voice():
    if False:
        print('Hello World!')
    voice = 'AwACAgIAAx0CAAGgr9AAAgmUX7b4c1KQyHVwzffxC2EnSYWsMAQAAgkAA4GkuUlsZUZ4_I97AR4E'
    voice_unique = 'AgADCQADgaS5SQ'
    check(voice, FileType.VOICE)
    check_unique(voice_unique, FileUniqueType.DOCUMENT)

def test_video_note():
    if False:
        i = 10
        return i + 15
    video_note = 'DQACAgIAAx0CAAGgr9AAAgmVX7b53qrRzCEO13BaLQJaYuFbdlwAAgoAA4GkuUmlqIzDy_PCsx4E'
    video_note_unique = 'AgADCgADgaS5SQ'
    video_note_thumb = 'AAMCAgADHQIAAaCv0AACCZVftvneqtHMIQ7XcFotAlpi4Vt2XAACCgADgaS5SaWojMPL88KzIH3qihAAAwEAB20AA_meAQABHgQ'
    video_note_thumb_unique = 'AQADIH3qihAAA_meAQAB'
    check(video_note, FileType.VIDEO_NOTE)
    check_unique(video_note_unique, FileUniqueType.DOCUMENT)
    check(video_note_thumb, FileType.THUMBNAIL)
    check_unique(video_note_thumb_unique, FileUniqueType.PHOTO)

def test_sticker():
    if False:
        print('Hello World!')
    sticker = 'CAACAgEAAx0CAAGgr9AAAgmWX7b6uFeLlhXEgYrM8pIbGaQKRQ0AAswBAALjeAQAAbeooNv_tb6-HgQ'
    sticker_unique = 'AgADzAEAAuN4BAAB'
    sticker_thumb = 'AAMCAQADHQIAAaCv0AACCZZftvq4V4uWFcSBiszykhsZpApFDQACzAEAAuN4BAABt6ig2_-1vr5gWNkpAAQBAAdtAAM0BQACHgQ'
    sticker_thumb_unique = 'AQADYFjZKQAENAUAAg'
    check(sticker, FileType.STICKER)
    check_unique(sticker_unique, FileUniqueType.DOCUMENT)
    check(sticker_thumb, FileType.THUMBNAIL)
    check_unique(sticker_thumb_unique, FileUniqueType.PHOTO)

def test_photo():
    if False:
        print('Hello World!')
    photo_small = 'AgACAgIAAx0CAAGgr9AAAgmZX7b7IPLRl8NcV3EJkzHwI1gwT-oAAq2nMRuBpLlJPJY-URZfhTkgfeqKEAADAQADAgADbQADAZ8BAAEeBA'
    photo_small_unique = 'AQADIH3qihAAAwGfAQAB'
    photo_medium = 'AgACAgIAAx0CAAGgr9AAAgmZX7b7IPLRl8NcV3EJkzHwI1gwT-oAAq2nMRuBpLlJPJY-URZfhTkgfeqKEAADAQADAgADeAADAp8BAAEeBA'
    photo_medium_unique = 'AQADIH3qihAAAwKfAQAB'
    photo_big = 'AgACAgIAAx0CAAGgr9AAAgmZX7b7IPLRl8NcV3EJkzHwI1gwT-oAAq2nMRuBpLlJPJY-URZfhTkgfeqKEAADAQADAgADeQAD_54BAAEeBA'
    photo_big_unique = 'AQADIH3qihAAA_-eAQAB'
    check(photo_small, FileType.PHOTO)
    check_unique(photo_small_unique, FileUniqueType.PHOTO)
    check(photo_medium, FileType.PHOTO)
    check_unique(photo_medium_unique, FileUniqueType.PHOTO)
    check(photo_big, FileType.PHOTO)
    check_unique(photo_big_unique, FileUniqueType.PHOTO)

def test_chat_photo():
    if False:
        for i in range(10):
            print('nop')
    user_photo_small = 'AQADAgADrKcxGylBBQAJIH3qihAAAwIAAylBBQAF7bDHYwABnc983KcAAh4E'
    user_photo_small_unique = 'AQADIH3qihAAA9ynAAI'
    user_photo_big = 'AQADAgADrKcxGylBBQAJIH3qihAAAwMAAylBBQAF7bDHYwABnc983qcAAh4E'
    user_photo_big_unique = 'AQADIH3qihAAA96nAAI'
    chat_photo_small = 'AQADAgATIH3qihAAAwIAA3t3-P______AAjhngEAAR4E'
    chat_photo_small_unique = 'AQADIH3qihAAA-GeAQAB'
    chat_photo_big = 'AQADAgATIH3qihAAAwMAA3t3-P______AAjjngEAAR4E'
    chat_photo_big_unique = 'AQADIH3qihAAA-OeAQAB'
    channel_photo_small = 'AQADAgATIH3qihAAAwIAA-fFwCoX____MvARg8nvpc3RpwACHgQ'
    channel_photo_small_unique = 'AQADIH3qihAAA9GnAAI'
    channel_photo_big = 'AQADAgATIH3qihAAAwMAA-fFwCoX____MvARg8nvpc3TpwACHgQ'
    channel_photo_big_unique = 'AQADIH3qihAAA9OnAAI'
    check(user_photo_small, FileType.CHAT_PHOTO)
    check_unique(user_photo_small_unique, FileUniqueType.PHOTO)
    check(user_photo_big, FileType.CHAT_PHOTO)
    check_unique(user_photo_big_unique, FileUniqueType.PHOTO)
    check(chat_photo_small, FileType.CHAT_PHOTO)
    check_unique(chat_photo_small_unique, FileUniqueType.PHOTO)
    check(chat_photo_big, FileType.CHAT_PHOTO)
    check_unique(chat_photo_big_unique, FileUniqueType.PHOTO)
    check(channel_photo_small, FileType.CHAT_PHOTO)
    check_unique(channel_photo_small_unique, FileUniqueType.PHOTO)
    check(channel_photo_big, FileType.CHAT_PHOTO)
    check_unique(channel_photo_big_unique, FileUniqueType.PHOTO)

def test_old_file_id():
    if False:
        for i in range(10):
            print('nop')
    old = 'BQADBAADQNKSZqjl5DcROGn_eu5JtgAEAgAEAg'
    check(old, FileType.DOCUMENT)

def test_unknown_file_type():
    if False:
        return 10
    unknown = 'RQACAgIAAx0CAAGgr9AAAgmPX7b4UxbjNoFEO_L0I4s6wrXNJA8AAgQAA4GkuUm9FFvIaOhXWR4E'
    with pytest.raises(ValueError, match='Unknown file_type \\d+ of file_id \\w+'):
        check(unknown, FileType.DOCUMENT)

def test_unknown_thumbnail_source():
    if False:
        i = 10
        return i + 15
    unknown = 'AAMCAgADHQIAAaCv0AACCY9ftvhTFuM2gUQ78vQjizrCtc0kDwACBAADgaS5Sb0UW8ho6FdZIH3qihAAA6QBAAIeBA'
    with pytest.raises(ValueError, match='Unknown thumbnail_source \\d+ of file_id \\w+'):
        check(unknown, FileType.THUMBNAIL)

def test_stringify_file_id():
    if False:
        print('Hello World!')
    file_id = 'BQACAgIAAx0CAAGgr9AAAgmPX7b4UxbjNoFEO_L0I4s6wrXNJA8AAgQAA4GkuUm9FFvIaOhXWR4E'
    string = "{'major': 4, 'minor': 30, 'file_type': <FileType.DOCUMENT: 5>, 'dc_id': 2, 'file_reference': b'\\x02\\x00\\xa0\\xaf\\xd0\\x00\\x00\\t\\x8f_\\xb6\\xf8S\\x16\\xe36\\x81D;\\xf2\\xf4#\\x8b:\\xc2\\xb5\\xcd$\\x0f', 'media_id': 5312458109417947140, 'access_hash': 6437869729085068477, 'thumbnail_size': ''}"
    assert str(FileId.decode(file_id)) == string

def test_stringify_file_unique_id():
    if False:
        while True:
            i = 10
    file_unique_id = 'AgADBAADgaS5SQ'
    string = "{'file_unique_type': <FileUniqueType.DOCUMENT: 2>, 'media_id': 5312458109417947140}"
    assert str(FileUniqueId.decode(file_unique_id)) == string