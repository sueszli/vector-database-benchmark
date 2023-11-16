from __future__ import annotations
import grp

def main():
    if False:
        i = 10
        return i + 15
    gids = [g.gr_gid for g in grp.getgrall()]
    i = 1
    while True:
        if i not in gids:
            print(i)
            break
        i += 1
if __name__ == '__main__':
    main()