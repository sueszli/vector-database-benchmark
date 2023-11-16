from abc import ABCMeta, abstractmethod

class Book(object, metaclass=ABCMeta):

    def __init__(self, title, author):
        if False:
            i = 10
            return i + 15
        self.title = title
        self.author = author

    @abstractmethod
    def display():
        if False:
            for i in range(10):
                print('nop')
        pass

class MyBook(Book):
    price = 0

    def __init__(self, title, author, price):
        if False:
            while True:
                i = 10
        super(Book, self).__init__()
        self.price = price

    def display(self):
        if False:
            print('Hello World!')
        print('Title: ' + title)
        print('Author: ' + author)
        print('Price: ' + str(price))
title = input()
author = input()
price = int(input())
new_novel = MyBook(title, author, price)
new_novel.display()