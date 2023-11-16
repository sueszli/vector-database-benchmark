"""Illustrates use of the :meth:`.AttributeEvents.init_scalar`
event, in conjunction with Core column defaults to provide
ORM objects that automatically produce the default value
when an un-set attribute is accessed.

"""
import datetime
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

def configure_listener(mapper, class_):
    if False:
        while True:
            i = 10
    'Establish attribute setters for every default-holding column on the\n    given mapper.'
    for col_attr in mapper.column_attrs:
        column = col_attr.columns[0]
        if column.default is not None:
            default_listener(col_attr, column.default)

def default_listener(col_attr, default):
    if False:
        print('Hello World!')
    'Establish a default-setting listener.\n\n    Given a class attribute and a :class:`.DefaultGenerator` instance.\n    The default generator should be a :class:`.ColumnDefault` object with a\n    plain Python value or callable default; otherwise, the appropriate behavior\n    for SQL functions and defaults should be determined here by the\n    user integrating this feature.\n\n    '

    @event.listens_for(col_attr, 'init_scalar', retval=True, propagate=True)
    def init_scalar(target, value, dict_):
        if False:
            return 10
        if default.is_callable:
            value = default.arg(None)
        elif default.is_scalar:
            value = default.arg
        else:
            raise NotImplementedError("Can't invoke pre-default for a SQL-level column default")
        dict_[col_attr.key] = value
        return value
if __name__ == '__main__':
    Base = declarative_base()
    event.listen(Base, 'mapper_configured', configure_listener, propagate=True)

    class Widget(Base):
        __tablename__ = 'widget'
        id = Column(Integer, primary_key=True)
        radius = Column(Integer, default=30)
        timestamp = Column(DateTime, default=datetime.datetime.now)
    e = create_engine('sqlite://', echo=True)
    Base.metadata.create_all(e)
    w1 = Widget()
    assert w1.radius == 30
    current_time = w1.timestamp
    assert current_time > datetime.datetime.now() - datetime.timedelta(seconds=5)
    sess = Session(e)
    sess.add(w1)
    sess.commit()
    assert sess.query(Widget.radius, Widget.timestamp).first() == (30, current_time)