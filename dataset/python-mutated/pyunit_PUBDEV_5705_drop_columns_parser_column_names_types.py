import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import random

def import_with_column_names_types():
    if False:
        return 10
    csvWithHeader = h2o.import_file(pyunit_utils.locate('smalldata/airlines/allyears2k_headers.zip'), header=1)
    allColnames = csvWithHeader.names
    allTypeDict = csvWithHeader.types
    csvWithNoHeader = h2o.upload_file(pyunit_utils.locate('smalldata/airlines/allyears2k.zip'), header=-1)
    allNewColnames = csvWithNoHeader.names
    allNewTypeDict = csvWithNoHeader.types
    pathNoHeader = pyunit_utils.locate('smalldata/airlines/allyears2k.zip')
    skip_even = list(range(0, csvWithHeader.ncol, 2))
    skip_odd = list(range(1, csvWithHeader.ncol, 2))
    skip_start_end = [0, csvWithHeader.ncol - 1]
    skip_except_last = list(range(0, csvWithHeader.ncol - 2))
    skip_except_first = list(range(1, csvWithHeader.ncol))
    temp = list(range(0, csvWithHeader.ncol))
    random.shuffle(temp)
    skip_random = []
    for index in range(0, csvWithHeader.ncol // 2):
        skip_random.append(temp[index])
    skip_random.sort()
    checkCorrectSkipAndNameTypes(csvWithHeader, pathNoHeader, skip_even, allColnames, allTypeDict, 0, 0)
    checkCorrectSkipAndNameTypes(csvWithNoHeader, pathNoHeader, skip_odd, allNewColnames, allNewTypeDict, 1, 0)
    checkCorrectSkipAndNameTypes(csvWithHeader, pathNoHeader, skip_start_end, allColnames, allTypeDict, 2, 0)
    checkCorrectSkipAndNameTypes(csvWithNoHeader, pathNoHeader, skip_except_last, allNewColnames, allNewTypeDict, 0, 0)
    checkCorrectSkipAndNameTypes(csvWithHeader, pathNoHeader, skip_except_first, allColnames, allTypeDict, 1, 0)
    checkCorrectSkipAndNameTypes(csvWithNoHeader, pathNoHeader, skip_random, allNewColnames, allNewTypeDict, 2, 0)

def checkCorrectSkipAndNameTypes(originalFullFrame, csvfile, skipped_columns, all_column_names, all_column_types, modes, headerValue):
    if False:
        i = 10
        return i + 15
    colnames = []
    coltypes = dict()
    coltypelist = []
    for ind in range(len(all_column_names)):
        if ind not in skipped_columns:
            colnames.append(all_column_names[ind])
            coltypes[all_column_names[ind]] = all_column_types[all_column_names[ind]]
            coltypelist.append(all_column_types[all_column_names[ind]])
    if modes == 0:
        skippedFrameUF = h2o.upload_file(csvfile, skipped_columns=skipped_columns, col_names=colnames, col_types=coltypes, header=headerValue)
        skippedFrameIF = h2o.import_file(csvfile, skipped_columns=skipped_columns, col_names=colnames, col_types=coltypes, header=headerValue)
    elif modes == 1:
        skippedFrameUF = h2o.upload_file(csvfile, skipped_columns=skipped_columns, col_names=colnames, header=headerValue)
        skippedFrameIF = h2o.import_file(csvfile, skipped_columns=skipped_columns, col_names=colnames, header=headerValue)
    else:
        skippedFrameUF = h2o.upload_file(csvfile, skipped_columns=skipped_columns, col_types=coltypelist, header=headerValue)
        skippedFrameIF = h2o.import_file(csvfile, skipped_columns=skipped_columns, col_types=coltypelist, header=headerValue)
    pyunit_utils.compare_frames_local(skippedFrameUF, skippedFrameIF, prob=0.5)
    skipCounter = 0
    typeDict = originalFullFrame.types
    frameNames = originalFullFrame.names
    for cindex in range(len(frameNames)):
        if cindex not in skipped_columns:
            print('Checking column {0}...'.format(cindex))
            if typeDict[frameNames[cindex]] == u'enum':
                pyunit_utils.compare_frames_local_onecolumn_NA_enum(originalFullFrame[cindex], skippedFrameIF[skipCounter], prob=1, tol=1e-10, returnResult=False)
            elif typeDict[frameNames[cindex]] == u'string':
                pyunit_utils.compare_frames_local_onecolumn_NA_string(originalFullFrame[cindex], skippedFrameIF[skipCounter], prob=1, returnResult=False)
            else:
                pyunit_utils.compare_frames_local_onecolumn_NA(originalFullFrame[cindex], skippedFrameIF[skipCounter], prob=1, tol=1e-10, returnResult=False)
            skipCounter = skipCounter + 1
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_with_column_names_types)
else:
    import_with_column_names_types()