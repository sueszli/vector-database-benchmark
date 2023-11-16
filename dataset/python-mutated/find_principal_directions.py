import numpy as np
import sklearn.svm
import multiprocessing as mp
indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]

def run(attrib_idx):
    if False:
        for i in range(10):
            print('nop')
    results = np.load('principal_directions/wspace_att_%d.npy' % attrib_idx).item()
    pruned_indices = list(range(results['latents'].shape[0]))
    svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
    space = 'dlatents'
    svm_inputs = results[space][pruned_indices]
    svm = sklearn.svm.LinearSVC(C=1.0, dual=False, max_iter=10000)
    svm.fit(svm_inputs, svm_targets)
    svm.score(svm_inputs, svm_targets)
    svm_outputs = svm.predict(svm_inputs)
    w = svm.coef_[0]
    np.save('principal_directions/direction_%d' % attrib_idx, w)
p = mp.Pool(processes=4)
p.map(run, indices)