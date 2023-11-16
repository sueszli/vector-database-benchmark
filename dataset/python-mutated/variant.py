class SemgrepVariant:

    def __init__(self, name: str, semgrep_core_extra: str, semgrep_extra: str=''):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.semgrep_core_extra = semgrep_core_extra
        self.semgrep_extra = semgrep_extra
from constants import STD
SEMGREP_VARIANTS = [SemgrepVariant(STD, ''), SemgrepVariant('no-gc-tuning', '-no_gc_tuning'), SemgrepVariant('no-filter-irrelevant-rules', '-no_filter_irrelevant_rules')]
STD_VARIANTS = [SemgrepVariant(STD, '')]