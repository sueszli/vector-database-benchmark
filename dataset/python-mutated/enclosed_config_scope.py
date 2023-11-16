from sacred.config.config_scope import ConfigScope
SIX = 6

@ConfigScope
def cfg():
    if False:
        for i in range(10):
            print('nop')
    answer = 7 * SIX

@ConfigScope
def cfg2():
    if False:
        i = 10
        return i + 15
    answer = 6 * SEVEN