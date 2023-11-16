# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *

from custom_data import *

class Test_CustomDataAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.AddData(Nifty, "NIFTY")
        self.AddData(CustomPythonData, "IBM", Resolution.Daily)

class Nifty(PythonData):
    '''NIFTY Custom Data Class'''
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource("https://www.dropbox.com/s/rsmg44jr6wexn2h/CNXNIFTY.csv?dl=1", SubscriptionTransportMedium.RemoteFile)


    def Reader(self, config, line, date, isLiveMode):
        if not (line.strip() and line[0].isdigit()): return None

        # New Nifty object
        index = Nifty()
        index.Symbol = config.Symbol

        try:
            # Example File Format:
            # Date,       Open       High        Low       Close     Volume      Turnover
            # 2011-09-13  7792.9    7799.9     7722.65    7748.7    116534670    6107.78
            data = line.split(',')
            index.Time = datetime.strptime(data[0], "%Y-%m-%d")
            index.Value = decimal.Decimal(data[4])
            index["Open"] = float(data[1])
            index["High"] = float(data[2])
            index["Low"] = float(data[3])
            index["Close"] = float(data[4])


        except ValueError:
                # Do nothing
                return None

        return index
