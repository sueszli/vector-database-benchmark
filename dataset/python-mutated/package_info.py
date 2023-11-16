"""(Legacy) Low-level implementation of a PackageRecord."""
from logging import getLogger
from ..auxlib.entity import ComposableField, Entity, EnumField, ImmutableEntity, IntegerField, ListField, StringField
from .channel import Channel
from .enums import NoarchType
from .records import PackageRecord, PathsData
log = getLogger(__name__)

class NoarchField(EnumField):

    def box(self, instance, instance_type, val):
        if False:
            while True:
                i = 10
        return super().box(instance, instance_type, NoarchType.coerce(val))

class Noarch(Entity):
    type = NoarchField(NoarchType)
    entry_points = ListField(str, required=False, nullable=True, default=None, default_in_dump=False)

class PreferredEnv(Entity):
    name = StringField()
    executable_paths = ListField(str, required=False, nullable=True)
    softlink_paths = ListField(str, required=False, nullable=True)

class PackageMetadata(Entity):
    package_metadata_version = IntegerField()
    noarch = ComposableField(Noarch, required=False, nullable=True)
    preferred_env = ComposableField(PreferredEnv, required=False, nullable=True, default=None, default_in_dump=False)

class PackageInfo(ImmutableEntity):
    extracted_package_dir = StringField()
    package_tarball_full_path = StringField()
    channel = ComposableField(Channel)
    repodata_record = ComposableField(PackageRecord)
    url = StringField()
    icondata = StringField(required=False, nullable=True)
    package_metadata = ComposableField(PackageMetadata, required=False, nullable=True)
    paths_data = ComposableField(PathsData)

    def dist_str(self):
        if False:
            for i in range(10):
                print('nop')
        return '{}::{}-{}-{}'.format(self.channel.canonical_name, self.name, self.version, self.build)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.repodata_record.name

    @property
    def version(self):
        if False:
            print('Hello World!')
        return self.repodata_record.version

    @property
    def build(self):
        if False:
            i = 10
            return i + 15
        return self.repodata_record.build

    @property
    def build_number(self):
        if False:
            for i in range(10):
                print('nop')
        return self.repodata_record.build_number