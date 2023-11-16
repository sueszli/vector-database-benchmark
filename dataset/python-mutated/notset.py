class NotSet:
    """Represents value that is not set.

    Can be used instead of the standard ``None`` in cases where ``None``
    itself is a valid value.

    Use the constant ``robot.utils.NOT_SET`` instead of creating new instances
    of the class.

    New in Robot Framework 7.0.
    """

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return ''
NOT_SET = NotSet()