"""`FactoryAggregate` provider example."""
import dataclasses
import sys
from dependency_injector import containers, providers

@dataclasses.dataclass
class Game:
    player1: str
    player2: str

    def play(self):
        if False:
            while True:
                i = 10
        print(f'{self.player1} and {self.player2} are playing {self.__class__.__name__.lower()}')

class Chess(Game):
    ...

class Checkers(Game):
    ...

class Ludo(Game):
    ...

class Container(containers.DeclarativeContainer):
    game_factory = providers.FactoryAggregate(chess=providers.Factory(Chess), checkers=providers.Factory(Checkers), ludo=providers.Factory(Ludo))
if __name__ == '__main__':
    game_type = sys.argv[1].lower()
    player1 = sys.argv[2].capitalize()
    player2 = sys.argv[3].capitalize()
    container = Container()
    selected_game = container.game_factory(game_type, player1, player2)
    selected_game.play()