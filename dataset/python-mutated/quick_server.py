import hug

@hug.get()
def quick():
    if False:
        i = 10
        return i + 15
    return 'Serving!'
if __name__ == '__main__':
    hug.API(__name__).http.serve()