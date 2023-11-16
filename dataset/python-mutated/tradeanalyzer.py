from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from backtrader import Analyzer
from backtrader.utils import AutoOrderedDict, AutoDict
from backtrader.utils.py3 import MAXINT

class TradeAnalyzer(Analyzer):
    """
    Provides statistics on closed trades (keeps also the count of open ones)

      - Total Open/Closed Trades

      - Streak Won/Lost Current/Longest

      - ProfitAndLoss Total/Average

      - Won/Lost Count/ Total PNL/ Average PNL / Max PNL

      - Long/Short Count/ Total PNL / Average PNL / Max PNL

          - Won/Lost Count/ Total PNL/ Average PNL / Max PNL

      - Length (bars in the market)

        - Total/Average/Max/Min

        - Won/Lost Total/Average/Max/Min

        - Long/Short Total/Average/Max/Min

          - Won/Lost Total/Average/Max/Min

    Note:

      The analyzer uses an "auto"dict for the fields, which means that if no
      trades are executed, no statistics will be generated.

      In that case there will be a single field/subfield in the dictionary
      returned by ``get_analysis``, namely:

        - dictname['total']['total'] which will have a value of 0 (the field is
          also reachable with dot notation dictname.total.total
    """

    def create_analysis(self):
        if False:
            i = 10
            return i + 15
        self.rets = AutoOrderedDict()
        self.rets.total.total = 0

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        super(TradeAnalyzer, self).stop()
        self.rets._close()

    def notify_trade(self, trade):
        if False:
            while True:
                i = 10
        if trade.justopened:
            self.rets.total.total += 1
            self.rets.total.open += 1
        elif trade.status == trade.Closed:
            trades = self.rets
            res = AutoDict()
            won = res.won = int(trade.pnlcomm >= 0.0)
            lost = res.lost = int(not won)
            tlong = res.tlong = trade.long
            tshort = res.tshort = not trade.long
            trades.total.open -= 1
            trades.total.closed += 1
            for wlname in ['won', 'lost']:
                wl = res[wlname]
                trades.streak[wlname].current *= wl
                trades.streak[wlname].current += wl
                ls = trades.streak[wlname].longest or 0
                trades.streak[wlname].longest = max(ls, trades.streak[wlname].current)
            trpnl = trades.pnl
            trpnl.gross.total += trade.pnl
            trpnl.gross.average = trades.pnl.gross.total / trades.total.closed
            trpnl.net.total += trade.pnlcomm
            trpnl.net.average = trades.pnl.net.total / trades.total.closed
            for wlname in ['won', 'lost']:
                wl = res[wlname]
                trwl = trades[wlname]
                trwl.total += wl
                trwlpnl = trwl.pnl
                pnlcomm = trade.pnlcomm * wl
                trwlpnl.total += pnlcomm
                trwlpnl.average = trwlpnl.total / (trwl.total or 1.0)
                wm = trwlpnl.max or 0.0
                func = max if wlname == 'won' else min
                trwlpnl.max = func(wm, pnlcomm)
            for tname in ['long', 'short']:
                trls = trades[tname]
                ls = res['t' + tname]
                trls.total += ls
                trls.pnl.total += trade.pnlcomm * ls
                trls.pnl.average = trls.pnl.total / (trls.total or 1.0)
                for wlname in ['won', 'lost']:
                    wl = res[wlname]
                    pnlcomm = trade.pnlcomm * wl * ls
                    trls[wlname] += wl * ls
                    trls.pnl[wlname].total += pnlcomm
                    trls.pnl[wlname].average = trls.pnl[wlname].total / (trls[wlname] or 1.0)
                    wm = trls.pnl[wlname].max or 0.0
                    func = max if wlname == 'won' else min
                    trls.pnl[wlname].max = func(wm, pnlcomm)
            trades.len.total += trade.barlen
            trades.len.average = trades.len.total / trades.total.closed
            ml = trades.len.max or 0
            trades.len.max = max(ml, trade.barlen)
            ml = trades.len.min or MAXINT
            trades.len.min = min(ml, trade.barlen)
            for wlname in ['won', 'lost']:
                trwl = trades.len[wlname]
                wl = res[wlname]
                trwl.total += trade.barlen * wl
                trwl.average = trwl.total / (trades[wlname].total or 1.0)
                m = trwl.max or 0
                trwl.max = max(m, trade.barlen * wl)
                if trade.barlen * wl:
                    m = trwl.min or MAXINT
                    trwl.min = min(m, trade.barlen * wl)
            for lsname in ['long', 'short']:
                trls = trades.len[lsname]
                ls = res['t' + lsname]
                barlen = trade.barlen * ls
                trls.total += barlen
                total_ls = trades[lsname].total
                trls.average = trls.total / (total_ls or 1.0)
                m = trls.max or 0
                trls.max = max(m, barlen)
                m = trls.min or MAXINT
                trls.min = min(m, barlen or m)
                for wlname in ['won', 'lost']:
                    wl = res[wlname]
                    barlen2 = trade.barlen * ls * wl
                    trls_wl = trls[wlname]
                    trls_wl.total += barlen2
                    trls_wl.average = trls_wl.total / (trades[lsname][wlname] or 1.0)
                    m = trls_wl.max or 0
                    trls_wl.max = max(m, barlen2)
                    m = trls_wl.min or MAXINT
                    trls_wl.min = min(m, barlen2 or m)