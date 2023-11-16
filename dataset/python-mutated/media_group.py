from typing import Any, Dict, List, Literal, Optional, Union, overload
from aiogram.enums import InputMediaType
from aiogram.types import UNSET_PARSE_MODE, InputFile, InputMedia, InputMediaAudio, InputMediaDocument, InputMediaPhoto, InputMediaVideo, MessageEntity
MediaType = Union[InputMediaAudio, InputMediaPhoto, InputMediaVideo, InputMediaDocument]
MAX_MEDIA_GROUP_SIZE = 10

class MediaGroupBuilder:

    def __init__(self, media: Optional[List[MediaType]]=None, caption: Optional[str]=None, caption_entities: Optional[List[MessageEntity]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper class for building media groups.\n\n        :param media: A list of media elements to add to the media group. (optional)\n        :param caption: Caption for the media group. (optional)\n        :param caption_entities: List of special entities in the caption,\n            like usernames, URLs, etc. (optional)\n        '
        self._media: List[MediaType] = []
        self.caption = caption
        self.caption_entities = caption_entities
        self._extend(media or [])

    def _add(self, media: MediaType) -> None:
        if False:
            return 10
        if not isinstance(media, InputMedia):
            raise ValueError('Media must be instance of InputMedia')
        if len(self._media) >= MAX_MEDIA_GROUP_SIZE:
            raise ValueError("Media group can't contain more than 10 elements")
        self._media.append(media)

    def _extend(self, media: List[MediaType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        for m in media:
            self._add(m)

    @overload
    def add(self, *, type: Literal[InputMediaType.AUDIO], media: Union[str, InputFile], caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, duration: Optional[int]=None, performer: Optional[str]=None, title: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def add(self, *, type: Literal[InputMediaType.PHOTO], media: Union[str, InputFile], caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, has_spoiler: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def add(self, *, type: Literal[InputMediaType.VIDEO], media: Union[str, InputFile], thumbnail: Optional[Union[InputFile, str]]=None, caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, width: Optional[int]=None, height: Optional[int]=None, duration: Optional[int]=None, supports_streaming: Optional[bool]=None, has_spoiler: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        pass

    @overload
    def add(self, *, type: Literal[InputMediaType.DOCUMENT], media: Union[str, InputFile], thumbnail: Optional[Union[InputFile, str]]=None, caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, disable_content_type_detection: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        pass

    def add(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add a media object to the media group.\n\n        :param kwargs: Keyword arguments for the media object.\n                The available keyword arguments depend on the media type.\n        :return: None\n        '
        type_ = kwargs.pop('type', None)
        if type_ == InputMediaType.AUDIO:
            self.add_audio(**kwargs)
        elif type_ == InputMediaType.PHOTO:
            self.add_photo(**kwargs)
        elif type_ == InputMediaType.VIDEO:
            self.add_video(**kwargs)
        elif type_ == InputMediaType.DOCUMENT:
            self.add_document(**kwargs)
        else:
            raise ValueError(f'Unknown media type: {type_!r}')

    def add_audio(self, media: Union[str, InputFile], thumbnail: Optional[Union[InputFile, str]]=None, caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, duration: Optional[int]=None, performer: Optional[str]=None, title: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            return 10
        "\n        Add an audio file to the media group.\n\n        :param media: File to send. Pass a file_id to send a file that exists on the\n            Telegram servers (recommended), pass an HTTP URL for Telegram to get a file from\n            the Internet, or pass 'attach://<file_attach_name>' to upload a new one using\n            multipart/form-data under <file_attach_name> name.\n             :ref:`More information on Sending Files » <sending-files>`\n        :param thumbnail: *Optional*. Thumbnail of the file sent; can be ignored if\n            thumbnail generation for the file is supported server-side. The thumbnail should\n            be in JPEG format and less than 200 kB in size. A thumbnail's width and height\n            should not exceed 320.\n        :param caption: *Optional*. Caption of the audio to be sent, 0-1024 characters\n            after entities parsing\n        :param parse_mode: *Optional*. Mode for parsing entities in the audio caption.\n            See `formatting options <https://core.telegram.org/bots/api#formatting-options>`_\n            for more details.\n        :param caption_entities: *Optional*. List of special entities that appear in the caption,\n            which can be specified instead of *parse_mode*\n        :param duration: *Optional*. Duration of the audio in seconds\n        :param performer: *Optional*. Performer of the audio\n        :param title: *Optional*. Title of the audio\n        :return: None\n        "
        self._add(InputMediaAudio(media=media, thumbnail=thumbnail, caption=caption, parse_mode=parse_mode, caption_entities=caption_entities, duration=duration, performer=performer, title=title, **kwargs))

    def add_photo(self, media: Union[str, InputFile], caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, has_spoiler: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Add a photo to the media group.\n\n        :param media: File to send. Pass a file_id to send a file that exists on the\n            Telegram servers (recommended), pass an HTTP URL for Telegram to get a file\n            from the Internet, or pass 'attach://<file_attach_name>' to upload a new\n            one using multipart/form-data under <file_attach_name> name.\n             :ref:`More information on Sending Files » <sending-files>`\n        :param caption: *Optional*. Caption of the photo to be sent, 0-1024 characters\n            after entities parsing\n        :param parse_mode: *Optional*. Mode for parsing entities in the photo caption.\n            See `formatting options <https://core.telegram.org/bots/api#formatting-options>`_\n            for more details.\n        :param caption_entities: *Optional*. List of special entities that appear in the caption,\n            which can be specified instead of *parse_mode*\n        :param has_spoiler: *Optional*. Pass :code:`True` if the photo needs to be covered\n            with a spoiler animation\n        :return: None\n        "
        self._add(InputMediaPhoto(media=media, caption=caption, parse_mode=parse_mode, caption_entities=caption_entities, has_spoiler=has_spoiler, **kwargs))

    def add_video(self, media: Union[str, InputFile], thumbnail: Optional[Union[InputFile, str]]=None, caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, width: Optional[int]=None, height: Optional[int]=None, duration: Optional[int]=None, supports_streaming: Optional[bool]=None, has_spoiler: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        "\n        Add a video to the media group.\n\n        :param media: File to send. Pass a file_id to send a file that exists on the\n            Telegram servers (recommended), pass an HTTP URL for Telegram to get a file\n            from the Internet, or pass 'attach://<file_attach_name>' to upload a new one\n            using multipart/form-data under <file_attach_name> name.\n            :ref:`More information on Sending Files » <sending-files>`\n        :param thumbnail: *Optional*. Thumbnail of the file sent; can be ignored if thumbnail\n            generation for the file is supported server-side. The thumbnail should be in JPEG\n            format and less than 200 kB in size. A thumbnail's width and height should\n            not exceed 320. Ignored if the file is not uploaded using multipart/form-data.\n            Thumbnails can't be reused and can be only uploaded as a new file, so you\n            can pass 'attach://<file_attach_name>' if the thumbnail was uploaded using\n            multipart/form-data under <file_attach_name>.\n            :ref:`More information on Sending Files » <sending-files>`\n        :param caption: *Optional*. Caption of the video to be sent,\n            0-1024 characters after entities parsing\n        :param parse_mode: *Optional*. Mode for parsing entities in the video caption.\n            See `formatting options <https://core.telegram.org/bots/api#formatting-options>`_\n            for more details.\n        :param caption_entities: *Optional*. List of special entities that appear in the caption,\n            which can be specified instead of *parse_mode*\n        :param width: *Optional*. Video width\n        :param height: *Optional*. Video height\n        :param duration: *Optional*. Video duration in seconds\n        :param supports_streaming: *Optional*. Pass :code:`True` if the uploaded video is\n            suitable for streaming\n        :param has_spoiler: *Optional*. Pass :code:`True` if the video needs to be covered\n            with a spoiler animation\n        :return: None\n        "
        self._add(InputMediaVideo(media=media, thumbnail=thumbnail, caption=caption, parse_mode=parse_mode, caption_entities=caption_entities, width=width, height=height, duration=duration, supports_streaming=supports_streaming, has_spoiler=has_spoiler, **kwargs))

    def add_document(self, media: Union[str, InputFile], thumbnail: Optional[Union[InputFile, str]]=None, caption: Optional[str]=None, parse_mode: Optional[str]=UNSET_PARSE_MODE, caption_entities: Optional[List[MessageEntity]]=None, disable_content_type_detection: Optional[bool]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        "\n        Add a document to the media group.\n\n        :param media: File to send. Pass a file_id to send a file that exists on the\n            Telegram servers (recommended), pass an HTTP URL for Telegram to get a file\n            from the Internet, or pass 'attach://<file_attach_name>' to upload a new one using\n            multipart/form-data under <file_attach_name> name.\n            :ref:`More information on Sending Files » <sending-files>`\n        :param thumbnail: *Optional*. Thumbnail of the file sent; can be ignored\n            if thumbnail generation for the file is supported server-side.\n            The thumbnail should be in JPEG format and less than 200 kB in size.\n            A thumbnail's width and height should not exceed 320.\n            Ignored if the file is not uploaded using multipart/form-data.\n            Thumbnails can't be reused and can be only uploaded as a new file,\n            so you can pass 'attach://<file_attach_name>' if the thumbnail was uploaded\n            using multipart/form-data under <file_attach_name>.\n            :ref:`More information on Sending Files » <sending-files>`\n        :param caption: *Optional*. Caption of the document to be sent,\n            0-1024 characters after entities parsing\n        :param parse_mode: *Optional*. Mode for parsing entities in the document caption.\n            See `formatting options <https://core.telegram.org/bots/api#formatting-options>`_\n            for more details.\n        :param caption_entities: *Optional*. List of special entities that appear\n            in the caption, which can be specified instead of *parse_mode*\n        :param disable_content_type_detection: *Optional*. Disables automatic server-side\n            content type detection for files uploaded using multipart/form-data.\n            Always :code:`True`, if the document is sent as part of an album.\n        :return: None\n\n        "
        self._add(InputMediaDocument(media=media, thumbnail=thumbnail, caption=caption, parse_mode=parse_mode, caption_entities=caption_entities, disable_content_type_detection=disable_content_type_detection, **kwargs))

    def build(self) -> List[MediaType]:
        if False:
            return 10
        '\n        Builds a list of media objects for a media group.\n\n        Adds the caption to the first media object if it is present.\n\n        :return: List of media objects.\n        '
        update_first_media: Dict[str, Any] = {'caption': self.caption}
        if self.caption_entities is not None:
            update_first_media['caption_entities'] = self.caption_entities
            update_first_media['parse_mode'] = None
        return [media.model_copy(update=update_first_media) if index == 0 and self.caption is not None else media for (index, media) in enumerate(self._media)]