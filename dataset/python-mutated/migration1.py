import logging
from google.appengine.api import memcache
from google.appengine.ext import ndb
import webapp2

class Person(ndb.Model):
    name = ndb.StringProperty(required=True)

def get_or_add_person(name):
    if False:
        for i in range(10):
            print('nop')
    person = memcache.get(name)
    if person is None:
        person = Person(name=name)
        memcache.add(name, person)
    else:
        logging.info('Found in cache: ' + name)
    return person

class MainPage(webapp2.RequestHandler):

    def get(self):
        if False:
            while True:
                i = 10
        person = get_or_add_person('Stevie Wonder')
        self.response.content_type = 'text/html'
        self.response.write(person.name)
app = webapp2.WSGIApplication([('/', MainPage)], debug=True)