"""Celko's "Nested Sets" Tree Structure.

https://www.intelligententerprise.com/001020/celko.jhtml

"""
from sqlalchemy import case
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import aliased
from sqlalchemy.orm import Session
Base = declarative_base()

class Employee(Base):
    __tablename__ = 'personnel'
    __mapper_args__ = {'batch': False}
    parent = None
    emp = Column(String, primary_key=True)
    left = Column('lft', Integer, nullable=False)
    right = Column('rgt', Integer, nullable=False)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Employee(%s, %d, %d)' % (self.emp, self.left, self.right)

@event.listens_for(Employee, 'before_insert')
def before_insert(mapper, connection, instance):
    if False:
        return 10
    if not instance.parent:
        instance.left = 1
        instance.right = 2
    else:
        personnel = mapper.mapped_table
        right_most_sibling = connection.scalar(select(personnel.c.rgt).where(personnel.c.emp == instance.parent.emp))
        connection.execute(personnel.update().where(personnel.c.rgt >= right_most_sibling).values(lft=case((personnel.c.lft > right_most_sibling, personnel.c.lft + 2), else_=personnel.c.lft), rgt=case((personnel.c.rgt >= right_most_sibling, personnel.c.rgt + 2), else_=personnel.c.rgt)))
        instance.left = right_most_sibling
        instance.right = right_most_sibling + 1
engine = create_engine('sqlite://', echo=True)
Base.metadata.create_all(engine)
session = Session(bind=engine)
albert = Employee(emp='Albert')
bert = Employee(emp='Bert')
chuck = Employee(emp='Chuck')
donna = Employee(emp='Donna')
eddie = Employee(emp='Eddie')
fred = Employee(emp='Fred')
bert.parent = albert
chuck.parent = albert
donna.parent = chuck
eddie.parent = chuck
fred.parent = chuck
session.add_all([albert, bert, chuck, donna, eddie, fred])
session.commit()
print(session.query(Employee).all())
ealias = aliased(Employee)
print(session.query(Employee).filter(ealias.left.between(Employee.left, Employee.right)).filter(ealias.emp == 'Eddie').all())
print(session.query(Employee).filter(Employee.left.between(ealias.left, ealias.right)).filter(ealias.emp == 'Chuck').all())
for (indentation, employee) in session.query(func.count(Employee.emp).label('indentation') - 1, ealias).filter(ealias.left.between(Employee.left, Employee.right)).group_by(ealias.emp).order_by(ealias.left):
    print('    ' * indentation + str(employee))