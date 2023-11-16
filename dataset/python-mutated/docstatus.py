class DocStatus(int):

    def is_draft(self):
        if False:
            print('Hello World!')
        return self == self.draft()

    def is_submitted(self):
        if False:
            return 10
        return self == self.submitted()

    def is_cancelled(self):
        if False:
            print('Hello World!')
        return self == self.cancelled()

    @classmethod
    def draft(cls):
        if False:
            return 10
        return cls(0)

    @classmethod
    def submitted(cls):
        if False:
            print('Hello World!')
        return cls(1)

    @classmethod
    def cancelled(cls):
        if False:
            return 10
        return cls(2)