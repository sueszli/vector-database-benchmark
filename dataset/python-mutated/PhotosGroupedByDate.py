import pytz
utc = pytz.UTC

class PhotosGroupedByDate:

    def __init__(self, location, date, photos):
        if False:
            return 10
        self.photos = photos
        self.date = date
        self.location = location

def get_photos_ordered_by_date(photos):
    if False:
        return 10
    from collections import defaultdict
    groups = defaultdict(list)
    for photo in photos:
        if photo.exif_timestamp:
            groups[photo.exif_timestamp.date().strftime('%Y-%m-%d')].append(photo)
        else:
            groups[photo.exif_timestamp].append(photo)
    groupedPhoto = list(groups.values())
    result = []
    noTimestampPhotos = []
    for group in groupedPhoto:
        location = ''
        if group[0].exif_timestamp:
            date = group[0].exif_timestamp.date().strftime('%Y-%m-%d')
            result.append(PhotosGroupedByDate(location, date, group))
        else:
            date = 'No timestamp'
            noTimestampPhotos = PhotosGroupedByDate(location, date, group)
    if noTimestampPhotos != []:
        result.append(noTimestampPhotos)
    return result