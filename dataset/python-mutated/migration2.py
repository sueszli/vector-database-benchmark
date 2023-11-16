import logging
from google.appengine.api import memcache
from google.appengine.ext import ndb
import webapp2

class Person(ndb.Model):
    name = ndb.StringProperty(required=True)
    userid = ndb.StringProperty(required=True)

def get_or_add_person(name, userid):
    if False:
        print('Hello World!')
    person = memcache.get(name)
    if person is None:
        person = Person(name=name, userid=userid)
        memcache.add(name, person)
    else:
        logging.info('Found in cache: ' + name + ', userid: ' + person.userid)
    return person

class MainPage(webapp2.RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        person = get_or_add_person('Stevie Wonder', '1')
        self.response.content_type = 'text/html'
        self.response.write(person.name)
app = webapp2.WSGIApplication([('/', MainPage)], debug=True)