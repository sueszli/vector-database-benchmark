"""ParentMgr module: contains the ParentMgr class"""
from direct.directnotify import DirectNotifyGlobal
from direct.showbase.PythonUtil import isDefaultValue

class ParentMgr:
    """ParentMgr holds a table of nodes that avatars may be parented to
    in a distributed manner. All clients within a particular zone maintain
    identical tables of these nodes, and the nodes are referenced by 'tokens'
    which the clients can pass to each other to communicate distributed
    reparenting information.

    The functionality of ParentMgr used to be implemented with a simple
    token->node dictionary. As distributed 'parent' objects were manifested,
    they would add themselves to the dictionary. Problems occured when
    distributed avatars were manifested before the objects to which they
    were parented to.

    Since the order of object manifestation depends on the order of the
    classes in the DC file, we could maintain an ordering of DC definitions
    that ensures that the necessary objects are manifested before avatars.
    However, it's easy enough to keep a list of pending reparents and thus
    support the general case without requiring any strict ordering in the DC.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('ParentMgr')

    def __init__(self):
        if False:
            print('Hello World!')
        self.token2nodepath = {}
        self.pendingParentToken2children = {}
        self.pendingChild2parentToken = {}

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        del self.token2nodepath
        del self.pendingParentToken2children
        del self.pendingChild2parentToken

    def privRemoveReparentRequest(self, child):
        if False:
            for i in range(10):
                print('nop')
        ' this internal function removes any currently-pending reparent\n        request for the given child nodepath '
        if child in self.pendingChild2parentToken:
            self.notify.debug("cancelling pending reparent of %s to '%s'" % (repr(child), self.pendingChild2parentToken[child]))
            parentToken = self.pendingChild2parentToken[child]
            del self.pendingChild2parentToken[child]
            self.pendingParentToken2children[parentToken].remove(child)

    def requestReparent(self, child, parentToken):
        if False:
            print('Hello World!')
        if parentToken in self.token2nodepath:
            self.privRemoveReparentRequest(child)
            self.notify.debug("performing wrtReparent of %s to '%s'" % (repr(child), parentToken))
            child.wrtReparentTo(self.token2nodepath[parentToken])
        else:
            if isDefaultValue(parentToken):
                self.notify.error('child %s requested reparent to default-value token: %s' % (repr(child), parentToken))
            self.notify.debug("child %s requested reparent to parent '%s' that is not (yet) registered" % (repr(child), parentToken))
            self.privRemoveReparentRequest(child)
            self.pendingChild2parentToken[child] = parentToken
            self.pendingParentToken2children.setdefault(parentToken, [])
            self.pendingParentToken2children[parentToken].append(child)
            child.reparentTo(hidden)

    def registerParent(self, token, parent):
        if False:
            while True:
                i = 10
        if token in self.token2nodepath:
            self.notify.error("registerParent: token '%s' already registered, referencing %s" % (token, repr(self.token2nodepath[token])))
        if isDefaultValue(token):
            self.notify.error('parent token (for %s) cannot be a default value (%s)' % (repr(parent), token))
        if isinstance(token, int):
            if token > 4294967295:
                self.notify.error('parent token %s (for %s) is out of uint32 range' % (token, repr(parent)))
        self.notify.debug("registering %s as '%s'" % (repr(parent), token))
        self.token2nodepath[token] = parent
        if token in self.pendingParentToken2children:
            children = self.pendingParentToken2children[token]
            del self.pendingParentToken2children[token]
            for child in children:
                self.notify.debug("performing reparent of %s to '%s'" % (repr(child), token))
                child.reparentTo(self.token2nodepath[token])
                assert self.pendingChild2parentToken[child] == token
                del self.pendingChild2parentToken[child]

    def unregisterParent(self, token):
        if False:
            i = 10
            return i + 15
        if token not in self.token2nodepath:
            self.notify.warning("unregisterParent: unknown parent token '%s'" % token)
            return
        self.notify.debug("unregistering parent '%s'" % token)
        del self.token2nodepath[token]