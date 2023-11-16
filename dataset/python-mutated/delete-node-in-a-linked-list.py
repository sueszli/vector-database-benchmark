class Solution(object):

    def deleteNode(self, node):
        if False:
            print('Hello World!')
        if node and node.next:
            node_to_delete = node.next
            node.val = node_to_delete.val
            node.next = node_to_delete.next
            del node_to_delete