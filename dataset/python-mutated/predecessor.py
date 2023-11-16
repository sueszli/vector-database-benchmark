def predecessor(root, node):
    if False:
        while True:
            i = 10
    pred = None
    while root:
        if node.val > root.val:
            pred = root
            root = root.right
        else:
            root = root.left
    return pred