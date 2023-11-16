from labjack import ljm
import argparse

parser = argparse.ArgumentParser(prog='Labjack T7 470010370 controller debug version',description='''Outputs a 384Hz sine wave that is used to control the Optotune Lens Driver 4. The sine wave is designed to output 2.5-4V on DAC0.''',epilog='''Version 0.1''')
parser.add_argument('-v',"--verbosity",help ='Increase verbosity',default=0, action="count")
parser.add_argument('-wvfrmres',type=int,help='Periods per 127 dots. Defaults to 3.',nargs='?',default=3)
parser.add_argument('-scantime',type=int,help='Waveforms scanned by each read function. Proportionnal to size of data packet sent. Defaults to 1.',default=1)
parser.add_argument('-acqtime',type=int,help='Number of times the read function is called.',default=1)
args = parser.parse_args()

fq = 384

if int(fq/float(args.wvfrmres)) != fq/float(args.wvfrmres):
  raise AssertionError('Waveform resolution not multiple of frequency')

#Waveform generation 
fqreal = int(fq/float(args.wvfrmres))
val = [ 3.25        ,3.36090803  ,3.46937734  ,3.57302286  ,3.66956556  ,3.7568826
  ,3.83305401  ,3.89640489  ,3.94554223  ,3.97938558  ,3.99719076  ,3.99856628
  ,3.98348187  ,3.95226924  ,3.90561469  ,3.8445441   ,3.77040032  ,3.68481367
  ,3.58966609  ,3.48704972  ,3.37922096  ,3.26855082  ,3.15747277  ,3.04842927
  ,2.94381802  ,2.84593928  ,2.75694526  ,2.67879283  ,2.61320044  ,2.56161037
  ,2.52515702  ,2.50464195  ,2.50051625  ,2.51287064  ,2.54143346  ,2.58557667
  ,2.64432961  ,2.71640039  ,2.80020427  ,2.89389853  ,2.99542295  ,3.10254517
  ,3.21290971  ,3.32408981  ,3.43364079  ,3.53915376  ,3.63830864  ,3.72892517
  ,3.80901081  ,3.87680459  ,3.93081581  ,3.96985686  ,3.99306927  ,3.99994263
  ,3.99032581  ,3.96443027  ,3.92282542  ,3.86642608  ,3.7964724   ,3.71450256
  ,3.62231897  ,3.5219486   ,3.41559847  ,3.30560706  ,3.19439294  ,3.08440153
  ,2.9780514   ,2.87768103  ,2.78549744  ,2.7035276   ,2.63357392  ,2.57717458
  ,2.53556973  ,2.50967419  ,2.50005737  ,2.50693073  ,2.53014314  ,2.56918419
  ,2.62319541  ,2.69098919  ,2.77107483  ,2.86169136  ,2.96084624  ,3.06635921
  ,3.17591019  ,3.28709029  ,3.39745483  ,3.50457705  ,3.60610147  ,3.69979573
  ,3.78359961  ,3.85567039  ,3.91442333  ,3.95856654  ,3.98712936  ,3.99948375
  ,3.99535805  ,3.97484298  ,3.93838963  ,3.88679956  ,3.82120717  ,3.74305474
  ,3.65406072  ,3.55618198  ,3.45157073  ,3.34252723  ,3.23144918  ,3.12077904
  ,3.01295028  ,2.91033391  ,2.81518633  ,2.72959968  ,2.6554559   ,2.59438531
  ,2.54773076  ,2.51651813  ,2.50143372  ,2.50280924  ,2.52061442  ,2.55445777
  ,2.60359511  ,2.66694599  ,2.7431174   ,2.83043444  ,2.92697714  ,3.03062266
  ,3.13909197  ,3.25      ]

#Paramters for acquisition
scanres = int(args.scantime * int(127))

data = []

#Opening labjack and generating handle object
handle = ljm.openS("T7","ETHERNET","470010370")

#Writing constants for Stream-Out
ljm.eWriteName(handle, "STREAM_OUT0_ENABLE",0)
ljm.eWriteName(handle,"STREAM_OUT0_TARGET",1000)
ljm.eWriteName(handle,"STREAM_OUT0_BUFFER_SIZE",256)
ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",1)

#Writing the waveform onto the labjack
ljm.eWriteName(handle,"STREAM_OUT0_LOOP_SIZE",127)
ljm.eWriteNameArray(handle,"STREAM_OUT0_BUFFER_F32",127,val[0:127])
ljm.eWriteName(handle,"STREAM_OUT0_SET_LOOP",1)

#Start streaming. 4800 is Stream-Out. 0 is Stream-In
ljm.eStreamStart(handle,scanres,2,[4800,2],fqreal*127)

for i in range(args.acqtime):
  read = ljm.eStreamRead(handle)
  data.append(read[0][0:scanres])
  if args.verbosity >0:
    print(ljm.eStreamRead(handle)[1:])

ljm.eStreamStop(handle)
ljm.close(handle)