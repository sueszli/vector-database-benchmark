import pulumi

def not_implemented(msg):
    if False:
        return 10
    raise NotImplementedError(msg)
pulumi.export('result', not_implemented('expression here is not implemented yet'))