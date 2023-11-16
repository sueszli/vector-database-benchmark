"""`Factory` provider init injections example."""
from dependency_injector import containers, providers

class Photo:
    ...

class User:

    def __init__(self, uid: int, main_photo: Photo) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.uid = uid
        self.main_photo = main_photo

class Container(containers.DeclarativeContainer):
    photo_factory = providers.Factory(Photo)
    user_factory = providers.Factory(User, main_photo=photo_factory)
if __name__ == '__main__':
    container = Container()
    user1 = container.user_factory(1)
    user2 = container.user_factory(2)
    another_photo = Photo()
    user3 = container.user_factory(uid=3, main_photo=another_photo)