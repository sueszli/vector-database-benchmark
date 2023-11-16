import numpy
from sklearn.model_selection import LeaveOneOut, LeavePOut
P_VAL = 2

def print_result(split_data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prints the result of either a LPOCV or LOOCV operation\n\n    Args:\n        split_data: The resulting (train, test) split data\n    '
    for (train, test) in split_data:
        output_train = ''
        output_test = ''
        bar = ['-'] * (len(train) + len(test))
        for i in train:
            output_train = '{}({}: {}) '.format(output_train, i, data[i])
        for i in test:
            bar[i] = 'T'
            output_test = '{}({}: {}) '.format(output_test, i, data[i])
        print('[ {} ]'.format(' '.join(bar)))
        print('Train: {}'.format(output_train))
        print('Test:  {}\n'.format(output_test))
data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])
loocv = LeaveOneOut()
lpocv = LeavePOut(p=P_VAL)
split_loocv = loocv.split(data)
split_lpocv = lpocv.split(data)
print('The Leave-P-Out method works by using every combination of P points as test data.\n\nThe following output shows the result of splitting some sample data by Leave-One-Out and Leave-P-Out methods.\nA bar displaying the current train-test split as well as the actual data points are displayed for each split.\nIn the bar, "-" is a training point and "T" is a test point.\n')
print('Data:\n{}\n'.format(data))
print('Leave-One-Out:\n')
print_result(split_loocv)
print('Leave-P-Out (where p = {}):\n'.format(P_VAL))
print_result(split_lpocv)