from typing import Any, List, OrderedDict
from scalene.scalene_statistics import Filename, LineNumber, ScaleneStatistics

class ScaleneLeakAnalysis:
    growth_rate_threshold = 0.01
    leak_reporting_threshold = 0.05

    @staticmethod
    def compute_leaks(growth_rate: float, stats: ScaleneStatistics, avg_mallocs: OrderedDict[LineNumber, float], fname: Filename) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        if growth_rate / 100 < ScaleneLeakAnalysis.growth_rate_threshold:
            return []
        leaks = []
        keys = list(stats.leak_score[fname].keys())
        for (index, item) in enumerate(stats.leak_score[fname].values()):
            allocs = item[0]
            frees = item[1]
            expected_leak = 1.0 - (frees + 1) / (allocs - frees + 2)
            if expected_leak >= 1.0 - ScaleneLeakAnalysis.leak_reporting_threshold:
                if keys[index] in avg_mallocs:
                    leaks.append((keys[index], expected_leak, avg_mallocs[keys[index]]))
        return leaks