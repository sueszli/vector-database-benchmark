from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

class ProxyDict:

    def __init__(self, parent, collection_name, childclass, keyname):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.collection_name = collection_name
        self.childclass = childclass
        self.keyname = keyname

    @property
    def collection(self):
        if False:
            i = 10
            return i + 15
        return getattr(self.parent, self.collection_name)

    def keys(self):
        if False:
            return 10
        descriptor = getattr(self.childclass, self.keyname)
        return [x[0] for x in self.collection.values(descriptor)]

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        x = self.collection.filter_by(**{self.keyname: key}).first()
        if x:
            return x
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if False:
            return 10
        try:
            existing = self[key]
            self.collection.remove(existing)
        except KeyError:
            pass
        self.collection.append(value)
engine = create_engine('sqlite://', echo=True)
Base = declarative_base(engine)

class Parent(Base):
    __tablename__ = 'parent'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    _collection = relationship('Child', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def child_map(self):
        if False:
            print('Hello World!')
        return ProxyDict(self, '_collection', Child, 'key')

class Child(Base):
    __tablename__ = 'child'
    id = Column(Integer, primary_key=True)
    key = Column(String(50))
    parent_id = Column(Integer, ForeignKey('parent.id'))

    def __repr__(self):
        if False:
            return 10
        return 'Child(key=%r)' % self.key
Base.metadata.create_all()
sess = sessionmaker()()
p1 = Parent(name='p1')
sess.add(p1)
print('\n---------begin setting nodes, autoflush occurs\n')
p1.child_map['k1'] = Child(key='k1')
p1.child_map['k2'] = Child(key='k2')
print('\n---------print keys - flushes first\n')
print(list(p1.child_map.keys()))
print("\n---------print 'k1' node\n")
print(p1.child_map['k1'])
print("\n---------update 'k2' node - must find existing, and replace\n")
p1.child_map['k2'] = Child(key='k2')
print("\n---------print 'k2' key - flushes first\n")
print(p1.child_map['k2'])
print('\n---------print all child nodes\n')
print(sess.query(Child).all())