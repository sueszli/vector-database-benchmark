def successor(root, node):
    if False:
        for i in range(10):
            print('nop')
    succ = None
    while root:
        if node.val < root.val:
            succ = root
            root = root.left
        else:
            root = root.right
    return succ