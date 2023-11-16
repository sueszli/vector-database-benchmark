import sys, os.path
from aubio import pvoc, source, float_type
from numpy import zeros, log10, vstack
import matplotlib.pyplot as plt

def get_spectrogram(filename, samplerate=0):
    if False:
        return 10
    win_s = 512
    hop_s = win_s // 2
    fft_s = win_s // 2 + 1
    a = source(filename, samplerate, hop_s)
    if samplerate == 0:
        samplerate = a.samplerate
    pv = pvoc(win_s, hop_s)
    specgram = zeros([0, fft_s], dtype=float_type)
    while True:
        (samples, read) = a()
        specgram = vstack((specgram, pv(samples).norm))
        if read < a.hop_size:
            break
    fig = plt.imshow(log10(specgram.T + 0.001), origin='bottom', aspect='auto', cmap=plt.cm.gray_r)
    ax = fig.axes
    ax.axis([0, len(specgram), 0, len(specgram[0])])
    time_step = hop_s / float(samplerate)
    total_time = len(specgram) * time_step
    outstr = 'total time: %0.2fs' % total_time
    print(outstr + ', samplerate: %.2fkHz' % (samplerate / 1000.0))
    n_xticks = 10
    n_yticks = 10

    def get_rounded_ticks(top_pos, step, n_ticks):
        if False:
            while True:
                i = 10
        top_label = top_pos * step
        ticks_first_label = top_pos * step / n_ticks
        ticks_first_label = round(ticks_first_label * 10.0) / 10.0
        ticks_labels = [ticks_first_label * n for n in range(n_ticks)] + [top_label]
        ticks_positions = [ticks_labels[n] / step for n in range(n_ticks)] + [top_pos]
        ticks_labels = ['%.1f' % x for x in ticks_labels]
        return (ticks_positions, ticks_labels)
    (x_ticks, x_labels) = get_rounded_ticks(len(specgram), time_step, n_xticks)
    (y_ticks, y_labels) = get_rounded_ticks(len(specgram[0]), samplerate / 1000.0 / 2.0 / len(specgram[0]), n_yticks)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(os.path.basename(filename))
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize('x-small')
    return fig
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s <filename>' % sys.argv[0])
    else:
        for soundfile in sys.argv[1:]:
            fig = get_spectrogram(soundfile)
            plt.show()
            plt.close()