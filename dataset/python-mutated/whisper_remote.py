import io
import logging
import os
from typing import Any, Dict, List, Optional
import openai
from haystack.preview import Document, component, default_from_dict, default_to_dict
from haystack.preview.dataclasses import ByteStream
logger = logging.getLogger(__name__)
API_BASE_URL = 'https://api.openai.com/v1'

@component
class RemoteWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper using OpenAI API. Requires an API key. See the
    [OpenAI blog post](https://beta.openai.com/docs/api-reference/whisper for more details.
    You can get one by signing up for an [OpenAI account](https://beta.openai.com/).

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text)
    """

    def __init__(self, api_key: Optional[str]=None, model_name: str='whisper-1', organization: Optional[str]=None, api_base_url: str=API_BASE_URL, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transcribes a list of audio files into a list of Documents.\n\n        :param api_key: OpenAI API key.\n        :param model_name: Name of the model to use. It now accepts only `whisper-1`.\n        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details, see OpenAI\n        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).\n        :param api_base: OpenAI base URL, defaults to `"https://api.openai.com/v1"`.\n        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the OpenAI\n            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio) for more details.\n            Some of the supported parameters:\n            - `language`: The language of the input audio.\n            Supplying the input language in ISO-639-1 format\n              will improve accuracy and latency.\n            - `prompt`: An optional text to guide the model\'s\n              style or continue a previous audio segment.\n              The prompt should match the audio language.\n            - `response_format`: The format of the transcript\n              output, in one of these options: json, text, srt,\n               verbose_json, or vtt. Defaults to "json". Currently only "json" is supported.\n            - `temperature`: The sampling temperature, between 0\n            and 1. Higher values like 0.8 will make the output more\n            random, while lower values like 0.2 will make it more\n            focused and deterministic. If set to 0, the model will\n            use log probability to automatically increase the\n            temperature until certain thresholds are hit.\n        '
        api_key = api_key or openai.api_key
        if api_key is None:
            try:
                api_key = os.environ['OPENAI_API_KEY']
            except KeyError as e:
                raise ValueError('RemoteWhisperTranscriber expects an OpenAI API key. Set the OPENAI_API_KEY environment variable (recommended) or pass it explicitly.') from e
        openai.api_key = api_key
        self.organization = organization
        self.model_name = model_name
        self.api_base_url = api_base_url
        whisper_params = kwargs
        if whisper_params.get('response_format') != 'json':
            logger.warning("RemoteWhisperTranscriber only supports 'response_format: json'. This parameter will be overwritten.")
        whisper_params['response_format'] = 'json'
        self.whisper_params = whisper_params
        if organization is not None:
            openai.organization = organization

    def to_dict(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Serialize this component to a dictionary.\n        This method overrides the default serializer in order to\n        avoid leaking the `api_key` value passed to the constructor.\n        '
        return default_to_dict(self, model_name=self.model_name, organization=self.organization, api_base_url=self.api_base_url, **self.whisper_params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemoteWhisperTranscriber':
        if False:
            print('Hello World!')
        '\n        Deserialize this component from a dictionary.\n        '
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, streams: List[ByteStream]):
        if False:
            print('Hello World!')
        '\n        Transcribe the audio files into a list of Documents, one for each input file.\n\n        For the supported audio formats, languages, and other parameters, see the\n        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper\n        [github repo](https://github.com/openai/whisper).\n\n        :param audio_files: a list of ByteStream objects to transcribe.\n        :returns: a list of Documents, one for each file. The content of the document is the transcription text.\n        '
        documents = []
        for stream in streams:
            file = io.BytesIO(stream.data)
            file.name = stream.metadata.get('file_path', 'audio_input.wav')
            content = openai.Audio.transcribe(file=file, model=self.model_name, **self.whisper_params)
            doc = Document(content=content['text'], meta=stream.metadata)
            documents.append(doc)
        return {'documents': documents}