def bug(state, slotstate):
    if False:
        return 10
    if state:
        if slotstate is not None:
            for (key, value) in slotstate.items():
                setattr(state, key, 2)