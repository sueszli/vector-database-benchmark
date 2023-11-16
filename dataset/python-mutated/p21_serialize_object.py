"""
Topic: 序列化一个对象
Desc : 
"""
import pickle

def serailize_object():
    if False:
        for i in range(10):
            print('nop')
    data = [1, 2, 3]
    f = open('somefile', 'wb')
    pickle.dump(data, f)
    s = pickle.dumps(data)
    f = open('somefile', 'rb')
    data = pickle.load(f)
    data = pickle.loads(s)
    f = open('somedata', 'wb')
    pickle.dump([1, 2, 3, 4], f)
    pickle.dump('hello', f)
    pickle.dump({'Apple', 'Pear', 'Banana'}, f)
    f.close()
    f = open('somedata', 'rb')
    print(pickle.load(f))
    print(pickle.load(f))
    print(pickle.load(f))
if __name__ == '__main__':
    serailize_object()