"""Base classes that are extended by low level AMQP frames and higher level
AMQP classes and methods.

"""

class AMQPObject:
    """Base object that is extended by AMQP low level frames and AMQP classes
    and methods.

    """
    NAME = 'AMQPObject'
    INDEX = None

    def __repr__(self):
        if False:
            print('Hello World!')
        items = list()
        for (key, value) in self.__dict__.items():
            if getattr(self.__class__, key, None) != value:
                items.append('{}={}'.format(key, value))
        if not items:
            return '<%s>' % self.NAME
        return '<{}({})>'.format(self.NAME, sorted(items))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other is not None:
            return self.__dict__ == other.__dict__
        else:
            return False

class Class(AMQPObject):
    """Is extended by AMQP classes"""
    NAME = 'Unextended Class'

class Method(AMQPObject):
    """Is extended by AMQP methods"""
    NAME = 'Unextended Method'
    synchronous = False

    def _set_content(self, properties, body):
        if False:
            for i in range(10):
                print('nop')
        'If the method is a content frame, set the properties and body to\n        be carried as attributes of the class.\n\n        :param pika.frame.Properties properties: AMQP Basic Properties\n        :param bytes body: The message body\n\n        '
        self._properties = properties
        self._body = body

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the properties if they are set.\n\n        :rtype: pika.frame.Properties\n\n        '
        return self._properties

    def get_body(self):
        if False:
            return 10
        'Return the message body if it is set.\n\n        :rtype: str|unicode\n\n        '
        return self._body

class Properties(AMQPObject):
    """Class to encompass message properties (AMQP Basic.Properties)"""
    NAME = 'Unextended Properties'