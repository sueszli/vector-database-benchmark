import math

class Solution(object):

    def printLinkedListInReverse(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: ImmutableListNode\n        :rtype: None\n        '

        def print_nodes(head, count):
            if False:
                while True:
                    i = 10
            nodes = []
            while head and len(nodes) != count:
                nodes.append(head)
                head = head.getNext()
            for node in reversed(nodes):
                node.printValue()
        count = 0
        curr = head
        while curr:
            curr = curr.getNext()
            count += 1
        bucket_count = int(math.ceil(count ** 0.5))
        buckets = []
        count = 0
        curr = head
        while curr:
            if count % bucket_count == 0:
                buckets.append(curr)
            curr = curr.getNext()
            count += 1
        for node in reversed(buckets):
            print_nodes(node, bucket_count)

class Solution2(object):

    def printLinkedListInReverse(self, head):
        if False:
            print('Hello World!')
        '\n        :type head: ImmutableListNode\n        :rtype: None\n        '
        nodes = []
        while head:
            nodes.append(head)
            head = head.getNext()
        for node in reversed(nodes):
            node.printValue()

class Solution3(object):

    def printLinkedListInReverse(self, head):
        if False:
            i = 10
            return i + 15
        '\n        :type head: ImmutableListNode\n        :rtype: None\n        '
        tail = None
        while head != tail:
            curr = head
            while curr.getNext() != tail:
                curr = curr.getNext()
            curr.printValue()
            tail = curr