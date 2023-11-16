def __getitem__(v):
    if False:
        return 10
    if v:
        try:
            return v
        except ValueError:
            try:
                return v
            except ValueError:
                pass
    return v