import json
d = dict(name='Bob', age=20, score=88)
data = json.dumps(d)
print('JSON Data is a str:', data)
reborn = json.loads(data)
print(reborn)

class Student(object):

    def __init__(self, name, age, score):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.age = age
        self.score = score

    def __str__(self):
        if False:
            return 10
        return 'Student object (%s, %s, %s)' % (self.name, self.age, self.score)
s = Student('Bob', 20, 88)
std_data = json.dumps(s, default=lambda obj: obj.__dict__)
print('Dump Student:', std_data)
rebuild = json.loads(std_data, object_hook=lambda d: Student(d['name'], d['age'], d['score']))
print(rebuild)