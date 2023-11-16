import numpy as np
import time
import matplotlib.pyplot as mpl
from scipy import signal
import argparse
from signalfunctions import *
from Rayoncomplx import *
parser = argparse.ArgumentParser(prog='Theoritical copy of Labjack controller T7')
parser.add_argument('-fq', type=int, help='Frequency of sine wave in Hz. Defaults to 385 Hz', nargs='?', default=120)
parser.add_argument('-wvfrmres', type=int, help='Periods per 127 dots. Defaults to 5.', nargs='?', default=2)
parser.add_argument('-scantime', type=float, help='Waveforms scanned by each read function. Proportionnal to size of data packet sent. Defaults to 2.', default=2)
parser.add_argument('-acqtime', type=int, help='Number of times the read function is called.', default=100)
parser.add_argument('-compar', help='Comparison between AIN0 and AIN1.', action='store_true', default=True)
args = parser.parse_args()
if int(args.fq / float(args.wvfrmres)) != args.fq / float(args.wvfrmres):
    raise AssertionError('Waveform resolution not multiple of frequency')
fqreal = int(args.fq / float(args.wvfrmres))
scanrate = fqreal * 127
scanres = int(args.scantime * int(127))
sp = np.linspace(0, int(args.wvfrmres) * 2 * np.pi * args.scantime, scanres + 1)
sp2 = np.linspace(0, 163 / float(60 * int(args.wvfrmres) * args.scantime) * 2 * np.pi * args.acqtime, (scanres + 1) * (args.acqtime + 3))

def width_of_beam(f, z):
    if False:
        print('Hello World!')
    return RayonComplexe(-0.2 + 10.1494j, 561).transform(syst2f(0.17) * lens(-0.17) * lens(f)).width(z)
val1 = 0.75 * np.sin(sp) + 3.25
focale = 0.3 * np.sin(sp) + 0.17
val2 = [1 / width_of_beam(i, 0) for i in focale]
data1 = np.array([])
data2 = np.array([])
for i in range(args.acqtime):
    read = (val1[0:scanres], val2[0:scanres])
    if args.compar == True or args.theo == True:
        if i > 0:
            data1 = np.append(data1, read[0])
            data2 = np.append(data2, read[1])
    time.sleep(1 / float(scanrate))
if args.compar == True:
    plotspace = np.linspace(0, (args.acqtime - 1) * args.scantime / float(fqreal), scanres * (args.acqtime - 1))
    plotspace2 = np.linspace(0, 1 / float(args.fq), 127 / args.wvfrmres)
    fig = mpl.figure()
    data3 = [1 if x >= np.mean(data1) else 0 for x in data1]
    data2norm = normalize_set(data2, data3)
    data2smooth = normalize_set(np.array([np.mean([data2[y + x * 127 / args.wvfrmres] for x in range(args.scantime * args.wvfrmres)]) for y in range(127 / args.wvfrmres)]), data3) + 0.5
    data2smoothmask = data2smooth > 0.1 * max(data2smooth)
    data2smoothpeaks = np.array([data2smooth[x] if data2smoothmask[x] == True else 0 for x in range(len(data2smooth))])
    peakind = signal.argrelmax(data2smoothpeaks)
    print(len(peakind[0]))
    if len(peakind[0]) == 2:
        data2shift = (peakind[0][1] - peakind[0][0]) * 1 / float(fqreal * 127) - 1 / float(2 * args.fq)
        print(data2shift)
    else:
        print('No shift calculated')
        data2shift = 0
    start = time.time()
    data4 = [1 if x in peakind[0] else 0 for x in range(len(data2smooth))]
    end = time.time()
    print(end - start)
    mpl.step(plotspace, data3)
    mpl.plot(plotspace, data2norm + 0.5)
    mpl.step(plotspace2 - data2shift + 1 / float(args.fq), data4)
    mpl.ylim(-0.5, 1.5)
    mpl.xlim(0, 3 / float(args.fq))
    mpl.show()
"\n\tfig.add_subplot(312)\n\t#Plot difference\n\tmpl.plot(plotspace,data3-data2shift)\n\t#mpl.plot(plotspace,data1-data2)\n\n\tfig.add_subplot(313)\n\t#Spectrum plot\n\tspectrum = abs(np.fft.rfft(data1+data2shift))[1:]\n\tfreqseq = np.fft.rfftfreq(len(data1+data2shift)-1,1/float(scanrate))\n\tprint('Dominant frequencies:{0}'.format(find_dominantfq(freqseq,spectrum,scanrate)))\n\tmpl.plot(freqseq,spectrum)\n\tmpl.xlim([args.fq-(1*args.fq),args.fq+(1*args.fq)])\n\t\n\tmpl.show()\n"