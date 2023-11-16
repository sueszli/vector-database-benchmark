import atexit

def orchest_notice():
    if False:
        for i in range(10):
            print('nop')
    print("\n \\ \\        / /             (_)\n  \\ \\  /\\  / /_ _ _ __ _ __  _ _ __   __ _\n   \\ \\/  \\/ / _` | '__| '_ \\| | '_ \\ / _` |\n    \\  /\\  / (_| | |  | | | | | | | | (_| |\n     \\/  \\/ \\__,_|_|  |_| |_|_|_| |_|\\__, |\n                                      __/ |\n                                     |___/\n\n# Please use Orchest environments to install pip packages.\n# NOTE: This only applies to installing packages inside Jupyter\n# kernels, not when installing Jupyter extensions.\n")
atexit.register(orchest_notice)