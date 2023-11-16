import os

def make_noise():
    if False:
        return 10
    'Make noise after finishing executing a code'
    duration = 1
    freq = 440
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def main():
    if False:
        return 10
    even_arr = []
    for i in range(10000):
        if i % 2 == 0:
            even_arr.append(i)
    make_noise()
if __name__ == '__main__':
    main()