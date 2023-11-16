from time import sleep

def main():
    if False:
        return 10
    import pyi_splash
    sleep(1)
    pyi_splash.update_text('This is a test text')
    sleep(2)
    pyi_splash.update_text("Second time's a charm")
    sleep(1)
    pyi_splash.close()
    sleep(20)
if __name__ == '__main__':
    main()