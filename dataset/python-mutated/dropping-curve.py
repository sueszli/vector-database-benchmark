import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from yellowbrick.datasets import load_game
from yellowbrick.model_selection.dropping_curve import dropping_curve

def main():
    if False:
        print('Hello World!')
    (X, y) = load_game()
    print(f'X.shape={X.shape}')
    print(f'y.shape={y.shape}')
    X_enc = OneHotEncoder().fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    (fig, ax) = plt.subplots()
    dropping_curve(MultinomialNB(), X_enc, y_enc, feature_sizes=np.linspace(0.05, 1, 20), ax=ax)
if __name__ == '__main__':
    main()