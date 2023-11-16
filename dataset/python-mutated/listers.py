"""Movie listers module."""
from .finders import MovieFinder

class MovieLister:

    def __init__(self, movie_finder: MovieFinder):
        if False:
            for i in range(10):
                print('nop')
        self._movie_finder = movie_finder

    def movies_directed_by(self, director):
        if False:
            return 10
        return [movie for movie in self._movie_finder.find_all() if movie.director == director]

    def movies_released_in(self, year):
        if False:
            for i in range(10):
                print('nop')
        return [movie for movie in self._movie_finder.find_all() if movie.year == year]