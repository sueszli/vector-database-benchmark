"""
To test this script directly you will need to import QuantConnect Dlls using clrloader from the appropriate 
location, the code below shows how to do this. Otherwise you can run it directly from C# in Lean without it.

To run as a solo script, add the following code to your script

Requires:
clr-loader==0.1.6
pandas

*********** CODE ***********
import os
import sys

# Get to DLL location where we are testing, change as needed
fileDirectory = os.path.dirname(os.path.abspath(__file__))
dlldir = "../../bin/Debug"
dlldir = os.path.join(fileDirectory, dlldir)

# Move us to dll directory and add it to path
os.chdir(dlldir)
sys.path.append(dlldir)

# Tell PythonNet to use .dotnet 6
from pythonnet import set_runtime
import clr_loader
set_runtime(clr_loader.get_coreclr(os.path.join(dlldir, "QuantConnect.Lean.Launcher.runtimeconfig.json")))

"""
from clr import AddReference
AddReference('QuantConnect.Common')
AddReference('QuantConnect.Tests')
from QuantConnect import *
from QuantConnect.Python import PandasConverter
from QuantConnect.Tests import Symbols
from QuantConnect.Tests.Python import PythonTestingUtils
import PandasMapper
import pandas as pd
spy = Symbols.SPY
aapl = Symbols.AAPL
SymbolCache.Set('SPY', spy)
SymbolCache.Set('AAPL', aapl)
pdConverter = PandasConverter()
slices = PythonTestingUtils.GetSlices(spy)
spydf = pdConverter.GetDataFrame(slices)
slices = PythonTestingUtils.GetSlices(aapl)
aapldf = pdConverter.GetDataFrame(slices)

def Test_Concat(dataFrame, dataFrame2, indexer):
    if False:
        i = 10
        return i + 15
    newDataFrame = pd.concat([dataFrame, dataFrame2])
    data = newDataFrame['lastprice'].unstack(level=0).iloc[-1][indexer]
    if data is 0:
        raise Exception('Data is zero')

def Test_Join(dataFrame, dataFrame2, indexer):
    if False:
        print('Hello World!')
    newDataFrame = dataFrame.join(dataFrame2, lsuffix='_')
    base = newDataFrame['lastprice_'].unstack(level=0)
    data = base.iloc[-1][indexer]
    if data is 0:
        raise Exception('Data is zero')
Test_Concat(spydf, aapldf, 'spy')
Test_Concat(spydf, aapldf, spy)
Test_Concat(spydf, aapldf, str(spy.ID))
Test_Join(spydf, aapldf, 'spy')
Test_Join(spydf, aapldf, spy)
Test_Join(spydf, aapldf, str(spy.ID))