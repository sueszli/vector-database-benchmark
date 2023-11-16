from telegram import PhotoSize, UserProfilePhotos
from tests.auxil.slots import mro_slots

class TestUserProfilePhotosBase:
    total_count = 2
    photos = [[PhotoSize('file_id1', 'file_un_id1', 512, 512), PhotoSize('file_id2', 'file_un_id1', 512, 512)], [PhotoSize('file_id3', 'file_un_id3', 512, 512), PhotoSize('file_id4', 'file_un_id4', 512, 512)]]

class TestUserProfilePhotosWithoutRequest(TestUserProfilePhotosBase):

    def test_slot_behaviour(self):
        if False:
            return 10
        inst = UserProfilePhotos(self.total_count, self.photos)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            while True:
                i = 10
        json_dict = {'total_count': 2, 'photos': [[y.to_dict() for y in x] for x in self.photos]}
        user_profile_photos = UserProfilePhotos.de_json(json_dict, bot)
        assert user_profile_photos.api_kwargs == {}
        assert user_profile_photos.total_count == self.total_count
        assert user_profile_photos.photos == tuple((tuple(p) for p in self.photos))

    def test_to_dict(self):
        if False:
            i = 10
            return i + 15
        user_profile_photos = UserProfilePhotos(self.total_count, self.photos)
        user_profile_photos_dict = user_profile_photos.to_dict()
        assert user_profile_photos_dict['total_count'] == user_profile_photos.total_count
        for (ix, x) in enumerate(user_profile_photos_dict['photos']):
            for (iy, y) in enumerate(x):
                assert y == user_profile_photos.photos[ix][iy].to_dict()

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = UserProfilePhotos(2, self.photos)
        b = UserProfilePhotos(2, self.photos)
        c = UserProfilePhotos(1, [self.photos[0]])
        d = PhotoSize('file_id1', 'unique_id', 512, 512)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)