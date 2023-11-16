"""
# A Guide to VisiData Guides
Each guide shows you how to use a particular feature in VisiData.

  [:keys]Up/Down[/] to move the row cursor
  [:keys]Enter[/] to view a topic
  [:keys]Backspace[/] to come back to this list of guides
"""
import re
from visidata import vd, BaseSheet, Sheet, ItemColumn, Column, VisiData

@VisiData.api
class GuideGuide(Sheet):
    help = __doc__
    columns = [ItemColumn('n', 0, type=int), ItemColumn('sheetname', 1, width=0), ItemColumn('topic', 2, width=60), Column('points', type=int, getter=lambda c, r: 0), Column('max_points', type=int, getter=lambda c, r: 100)]

    def iterload(self):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        for line in '\nGuideGuide ("A Guide to VisiData Guides (you are here)")\nHelpGuide ("Where to Start and How to Quit")  # manpage; ask for patreon\nMenuGuide ("The VisiData Menu System")\nCommandsSheet ("How to find the command you want run")\n\n#  real barebones basics\nMovementGuide ("Movement and Search")\nSortGuide ("Sorting")\nTypesSheet ("The basic type system")\nCommandLog  ("Undo and Replay")\n\n#  rev this thing up\n\nSelectionGuide ("Selecting and filtering") # stu|, and variants; filtering with dup-sheet; g prefix often refers to selected rowset\nSheetsSheet  ("The Sheet Stack")\nColumnsSheet ("Columns: the only way to fly")\nStatusesSheet ("Revisit old status messages")\nSidebarSheet ("Dive into the sidebar")\nSaversGuide ("Saving Data")\n\nErrorsSheet ("What was that error?")\nModifyGuide ("Adding, Editing, Deleting Rows")\n\n# the varieties of data experience\n\nSlideGuide ("Sliding rows and columns around")\nExprGuide ("Compute Python over every row")\nJoinGuide ("Joining multiple sheets together")\nDescribeSheet ("Basic Statistics (min/max/mode/median/mean)")\nAggregatorsSheet ("Aggregations like sum, mean, and ")\nFrequencyTable ("Frequency Tables are how you GROUP BY")\nPivotGuide ("Pivot Tables are just Frequency Tables with more columns")\nMeltGuide ("Melt is just Unpivot")\nJsonGuide ("Some special features for JSON") # with expand/contract, unfurl\nRegexGuide ("Matching and Transforming Strings with Regex")\nGraphSheet ("Basic scatterplots and other graphs")\n\n# for the frequent user\nOptionsSheet ("Options and Settings")\nClipboardGuide ("Copy and Paste Data via the Clipboard")\nDirSheet ("Browsing the local filesystem")\nFormatsSheet ("What can you open with VisiData?")\nThemesSheet ("Change Interface Theme")\nColorSheet ("See available colors")\nMacrosSheet ("Recording macros")\nMemorySheet ("Making note of certain values")\n\n# advanced usage and developers\n\nThreadsSheet ("Threads past and present")\nPyobjSheet ("Inspecting internal Python objects")\n\n#  appendices\n\nInputEditorGuide ("Using the builtin line editor")\n        '.splitlines():
            m = re.search('(\\w+?) \\("(.*)"\\)', line)
            if m:
                yield ([i] + list(m.groups()))
                i += 1
BaseSheet.addCommand('', 'open-guide', 'vd.push(GuideGuide("VisiData_Guide"))')