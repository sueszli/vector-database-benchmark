def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    return True

def get_tag_descriptions():
    if False:
        i = 10
        return i + 15

    class TagDescriptions(dict):
        """
        A TagDescriptions is used to collect the tags associated with ELB
        resources.
        See :class:`boto.ec2.elb.LoadBalancer` for more details.
        """

        def __init__(self, connection=None):
            if False:
                print('Hello World!')
            dict.__init__(self)
            self.connection = connection
            self._load_balancer_name = None
            self._tags = None

        def startElement(self, name, attrs, connection):
            if False:
                while True:
                    i = 10
            if name == 'member':
                self.load_balancer_name = None
                self.tags = None
            if name == 'Tags':
                self._tags = TagSet()
                return self._tags
            return None

        def endElement(self, name, value, connection):
            if False:
                print('Hello World!')
            if name == 'LoadBalancerName':
                self._load_balancer_name = value
            elif name == 'member':
                self[self._load_balancer_name] = self._tags

    class TagSet(dict):
        """
        A TagSet is used to collect the tags associated with a particular
        ELB resource.  See :class:`boto.ec2.elb.LoadBalancer` for more
        details.
        """

        def __init__(self, connection=None):
            if False:
                while True:
                    i = 10
            dict.__init__(self)
            self.connection = connection
            self._current_key = None
            self._current_value = None

        def startElement(self, name, attrs, connection):
            if False:
                for i in range(10):
                    print('nop')
            if name == 'member':
                self._current_key = None
                self._current_value = None
            return None

        def endElement(self, name, value, connection):
            if False:
                i = 10
                return i + 15
            if name == 'Key':
                self._current_key = value
            elif name == 'Value':
                self._current_value = value
            elif name == 'member':
                self[self._current_key] = self._current_value
    return TagDescriptions