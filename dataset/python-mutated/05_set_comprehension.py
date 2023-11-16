{y for y in range(3)}
b = {v: k for (k, v) in enumerate(b3)}

def __new__(classdict):
    if False:
        return 10
    members = {k: classdict[k] for k in classdict._member_names}
    return members
{a for b in bases for a in b.__dict__}