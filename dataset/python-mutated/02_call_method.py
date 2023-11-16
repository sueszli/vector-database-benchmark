def truncate(self, size=None):
    if False:
        print('Hello World!')
    self.db.put(self.key, '', txn=self.txn, dlen=self.len - size, doff=size)