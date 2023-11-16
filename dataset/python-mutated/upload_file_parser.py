import base64
import hashlib
import hmac
import logging
import os
import time
from typing import Optional
from flask import current_app
from extensions.ext_storage import storage
SUPPORT_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'gif']

class UploadFileParser:

    @classmethod
    def get_image_data(cls, upload_file, force_url: bool=False) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if not upload_file:
            return None
        if upload_file.extension not in SUPPORT_EXTENSIONS:
            return None
        if current_app.config['MULTIMODAL_SEND_IMAGE_FORMAT'] == 'url' or force_url:
            return cls.get_signed_temp_image_url(upload_file)
        else:
            try:
                data = storage.load(upload_file.key)
            except FileNotFoundError:
                logging.error(f'File not found: {upload_file.key}')
                return None
            encoded_string = base64.b64encode(data).decode('utf-8')
            return f'data:{upload_file.mime_type};base64,{encoded_string}'

    @classmethod
    def get_signed_temp_image_url(cls, upload_file) -> str:
        if False:
            return 10
        '\n        get signed url from upload file\n\n        :param upload_file: UploadFile object\n        :return:\n        '
        base_url = current_app.config.get('FILES_URL')
        image_preview_url = f'{base_url}/files/{upload_file.id}/image-preview'
        timestamp = str(int(time.time()))
        nonce = os.urandom(16).hex()
        data_to_sign = f'image-preview|{upload_file.id}|{timestamp}|{nonce}'
        secret_key = current_app.config['SECRET_KEY'].encode()
        sign = hmac.new(secret_key, data_to_sign.encode(), hashlib.sha256).digest()
        encoded_sign = base64.urlsafe_b64encode(sign).decode()
        return f'{image_preview_url}?timestamp={timestamp}&nonce={nonce}&sign={encoded_sign}'

    @classmethod
    def verify_image_file_signature(cls, upload_file_id: str, timestamp: str, nonce: str, sign: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        verify signature\n\n        :param upload_file_id: file id\n        :param timestamp: timestamp\n        :param nonce: nonce\n        :param sign: signature\n        :return:\n        '
        data_to_sign = f'image-preview|{upload_file_id}|{timestamp}|{nonce}'
        secret_key = current_app.config['SECRET_KEY'].encode()
        recalculated_sign = hmac.new(secret_key, data_to_sign.encode(), hashlib.sha256).digest()
        recalculated_encoded_sign = base64.urlsafe_b64encode(recalculated_sign).decode()
        if sign != recalculated_encoded_sign:
            return False
        current_time = int(time.time())
        return current_time - int(timestamp) <= 300