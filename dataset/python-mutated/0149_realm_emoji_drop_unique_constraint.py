import os
import shutil
import boto3.session
from django.conf import settings
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from mypy_boto3_s3.type_defs import CopySourceTypeDef
from typing_extensions import override

class Uploader:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.old_orig_image_path_template = '{realm_id}/emoji/{emoji_file_name}.original'
        self.old_path_template = '{realm_id}/emoji/{emoji_file_name}'
        self.new_orig_image_path_template = '{realm_id}/emoji/images/{emoji_file_name}.original'
        self.new_path_template = '{realm_id}/emoji/images/{emoji_file_name}'

    def copy_files(self, src_path: str, dst_path: str) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def ensure_emoji_images(self, realm_id: int, old_filename: str, new_filename: str) -> None:
        if False:
            while True:
                i = 10
        old_file_path = self.old_orig_image_path_template.format(realm_id=realm_id, emoji_file_name=old_filename)
        new_file_path = self.new_orig_image_path_template.format(realm_id=realm_id, emoji_file_name=new_filename)
        self.copy_files(old_file_path, new_file_path)
        old_file_path = self.old_path_template.format(realm_id=realm_id, emoji_file_name=old_filename)
        new_file_path = self.new_path_template.format(realm_id=realm_id, emoji_file_name=new_filename)
        self.copy_files(old_file_path, new_file_path)

class LocalUploader(Uploader):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    @staticmethod
    def mkdirs(path: str) -> None:
        if False:
            while True:
                i = 10
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    @override
    def copy_files(self, src_path: str, dst_path: str) -> None:
        if False:
            while True:
                i = 10
        assert settings.LOCAL_UPLOADS_DIR is not None
        assert settings.LOCAL_AVATARS_DIR is not None
        assert settings.LOCAL_FILES_DIR is not None
        src_path = os.path.join(settings.LOCAL_AVATARS_DIR, src_path)
        self.mkdirs(src_path)
        dst_path = os.path.join(settings.LOCAL_AVATARS_DIR, dst_path)
        self.mkdirs(dst_path)
        shutil.copyfile(src_path, dst_path)

class S3Uploader(Uploader):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        session = boto3.session.Session(settings.S3_KEY, settings.S3_SECRET_KEY)
        self.bucket_name = settings.S3_AVATAR_BUCKET
        self.bucket = session.resource('s3', region_name=settings.S3_REGION, endpoint_url=settings.S3_ENDPOINT_URL).Bucket(self.bucket_name)

    @override
    def copy_files(self, src_key: str, dst_key: str) -> None:
        if False:
            print('Hello World!')
        source = CopySourceTypeDef(Bucket=self.bucket_name, Key=src_key)
        self.bucket.copy(CopySource=source, Key=dst_key)

def get_uploader() -> Uploader:
    if False:
        while True:
            i = 10
    if settings.LOCAL_UPLOADS_DIR is None:
        return S3Uploader()
    return LocalUploader()

def get_emoji_file_name(emoji_file_name: str, new_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    (_, image_ext) = os.path.splitext(emoji_file_name)
    return f'{new_name}{image_ext}'

def migrate_realm_emoji_image_files(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    RealmEmoji = apps.get_model('zerver', 'RealmEmoji')
    uploader = get_uploader()
    for realm_emoji in RealmEmoji.objects.all():
        old_file_name = realm_emoji.file_name
        new_file_name = get_emoji_file_name(old_file_name, str(realm_emoji.id))
        uploader.ensure_emoji_images(realm_emoji.realm_id, old_file_name, new_file_name)
        realm_emoji.file_name = new_file_name
        realm_emoji.save(update_fields=['file_name'])

def reversal(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    RealmEmoji = apps.get_model('zerver', 'RealmEmoji')
    for realm_emoji in RealmEmoji.objects.all():
        corrupt_file_name = realm_emoji.file_name
        correct_file_name = get_emoji_file_name(corrupt_file_name, realm_emoji.name)
        realm_emoji.file_name = correct_file_name
        realm_emoji.save(update_fields=['file_name'])

class Migration(migrations.Migration):
    dependencies = [('zerver', '0148_max_invites_forget_default')]
    operations = [migrations.AlterUniqueTogether(name='realmemoji', unique_together=set()), migrations.AlterField(model_name='realmemoji', name='file_name', field=models.TextField(db_index=True, null=True, blank=True)), migrations.RunPython(migrate_realm_emoji_image_files, reverse_code=reversal, elidable=True)]