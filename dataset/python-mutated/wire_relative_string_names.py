"""Wiring sample package."""

def wire_with_relative_string_names(container):
    if False:
        return 10
    container.wire(modules=['.module'], packages=['.package'])