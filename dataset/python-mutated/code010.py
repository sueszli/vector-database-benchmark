import datetime
from ninja import Schema, Path

class PathDate(Schema):
    year: int
    month: int
    day: int

    def value(self):
        if False:
            print('Hello World!')
        return datetime.date(self.year, self.month, self.day)

@api.get('/events/{year}/{month}/{day}')
def events(request, date: PathDate=Path(...)):
    if False:
        print('Hello World!')
    return {'date': date.value()}