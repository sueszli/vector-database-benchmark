from django.db import models
from api.models.photo import Photo
from api.models.user import User, get_deleted_user

class AlbumDate(models.Model):
    title = models.CharField(blank=True, null=True, max_length=512, db_index=True)
    date = models.DateField(db_index=True, null=True)
    photos = models.ManyToManyField(Photo)
    favorited = models.BooleanField(default=False, db_index=True)
    location = models.JSONField(blank=True, db_index=True, null=True)
    owner = models.ForeignKey(User, on_delete=models.SET(get_deleted_user), default=None)
    shared_to = models.ManyToManyField(User, related_name='album_date_shared_to')
    objects = models.Manager()

    class Meta:
        unique_together = ('date', 'owner')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.date) + ' (' + str(self.owner) + ')'

    def ordered_photos(self):
        if False:
            for i in range(10):
                print('nop')
        return self.photos.all().order_by('-exif_timestamp')

def get_or_create_album_date(date, owner):
    if False:
        while True:
            i = 10
    try:
        return AlbumDate.objects.get_or_create(date=date, owner=owner)[0]
    except AlbumDate.MultipleObjectsReturned:
        return AlbumDate.objects.filter(date=date, owner=owner).first()

def get_album_date(date, owner):
    if False:
        print('Hello World!')
    try:
        return AlbumDate.objects.get(date=date, owner=owner)
    except Exception:
        return None

def get_album_nodate(owner):
    if False:
        i = 10
        return i + 15
    return AlbumDate.objects.get_or_create(date=None, owner=owner)[0]