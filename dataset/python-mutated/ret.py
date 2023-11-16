"""
Module to integrate with the returner system and retrieve data sent to a salt returner
"""
import salt.loader

def get_jid(returner, jid):
    if False:
        while True:
            i = 10
    "\n    Return the information for a specified job id\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ret.get_jid redis 20421104181954700505\n    "
    returners = salt.loader.returners(__opts__, __salt__)
    return returners['{}.get_jid'.format(returner)](jid)

def get_fun(returner, fun):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return info about last time fun was called on each minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ret.get_fun mysql network.interfaces\n    "
    returners = salt.loader.returners(__opts__, __salt__)
    return returners['{}.get_fun'.format(returner)](fun)

def get_jids(returner):
    if False:
        while True:
            i = 10
    "\n    Return a list of all job ids\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ret.get_jids mysql\n    "
    returners = salt.loader.returners(__opts__, __salt__)
    return returners['{}.get_jids'.format(returner)]()

def get_minions(returner):
    if False:
        return 10
    "\n    Return a list of all minions\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ret.get_minions mysql\n    "
    returners = salt.loader.returners(__opts__, __salt__)
    return returners['{}.get_minions'.format(returner)]()