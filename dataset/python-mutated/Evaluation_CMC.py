import pandas as pd
from matplotlib import pyplot as plt

def return_first_one_prob(prob):
    if False:
        i = 10
        return i + 15
    for index in range(len(prob)):
        if prob[index] == 1.0:
            return index

def evaluate(dataset):
    if False:
        i = 10
        return i + 15
    x = pd.read_csv(dataset)
    real = x.REAL
    scores = x.SCORES
    for i in range(len(scores)):
        scores[i] = scores[i][1:-1]
        scores[i] = scores[i].split()
        for j in range(len(scores[i])):
            scores[i][j] = float(scores[i][j])
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] = (scores[i][j], j)
    ranks = len(scores[0])
    CMS = dict()
    c = 0
    for k in range(ranks):
        CMS[k + 1] = c
        for i in range(len(real)):
            s_scores = sorted(scores[i], reverse=True)
            if s_scores[k][1] == real[i]:
                CMS[k + 1] += 1
                c += 1
        CMS[k + 1] = CMS[k + 1] / len(real)
    prob = [0] + list(CMS.values())
    index_first_prob_one = return_first_one_prob(prob)
    plt.figure()
    plt.plot(list(range(ranks + 1)), prob)
    plt.plot(index_first_prob_one, prob[index_first_prob_one], 'x', label='Probability = 1.0 at rank ' + str(index_first_prob_one))
    plt.axvline(index_first_prob_one, color='r', linestyle=':', linewidth='1')
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ranks')
    plt.ylabel('Prob. of identification')
    plt.title('Cumulative Match Characteristic')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('plot/CMC/CMC.svg', dpi=1200)
    plt.clf()
    plt.figure()
    plt.plot(list(range(1, 6)), prob[1:6])
    plt.plot(list(range(1, 6)), prob[1:6], 'x')
    plt.ylim([0.853, 0.88])
    plt.xlim([0.8, 5.5])
    plt.xlabel('Ranks')
    plt.ylabel('Prob. of identification')
    plt.title('Cumulative Match Characteristic')
    plt.grid()
    plt.legend(loc='lower right')
    for (label, xi, yi) in zip(prob[1:6], list(range(1, 6)), prob[1:6]):
        plt.annotate('{:.3f}'.format(label), xy=(xi, yi), xytext=(0, -12), textcoords='offset points')
    plt.savefig('plot/CMC/CMC_at_rank_5.svg', dpi=1200)
    plt.clf()
    print('Score at rank 1 (Also Called Recognition Rate): ', CMS[1])
    print('Score at rank 2: ', CMS[2])
    print('Score at rank 3: ', CMS[3])
    print('Score at rank 3: ', CMS[4])
    print('Score at rank 5: ', CMS[5])
if __name__ == '__main__':
    evaluate('datasets/predictions_dataset.csv')