from utils import *
import time

def test_train_set(data):
    if False:
        i = 10
        return i + 15
    features = data.drop(columns=['label'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    labels = data['label']
    (feature_train, feature_test, label_train, label_test) = train_test_split(features, labels, test_size=0.1)
    return (feature_train, feature_test, label_train, label_test)

def train_test_valid_LDA(data):
    if False:
        for i in range(10):
            print('nop')
    (feature_train, feature_test, label_train, label_test) = test_train_set(data)
    model = LinearDiscriminantAnalysis()
    model.fit(feature_train, label_train)
    prediction = model.predict(feature_test)
    accuracy = accuracy_score(label_test, prediction)
    print(f'Accuracy LDA: {accuracy:.2f}')
    cm = confusion_matrix(label_test, prediction, labels=model.classes_)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm.astype('int')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

def train_test_valid_MLP(data, hlayers, neurons, maxIter, activation, dataset):
    if False:
        print('Hello World!')
    (feature_train, feature_test, label_train, label_test) = test_train_set(data)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(label_train)
    y_test_encoded = le.transform(label_test)
    if hlayers == 2:
        model = MLPClassifier(hidden_layer_sizes=(neurons, neurons), max_iter=maxIter, activation=activation, random_state=1)
    if hlayers == 4:
        model = MLPClassifier(hidden_layer_sizes=(neurons, neurons), max_iter=maxIter, activation=activation, random_state=1)
    if hlayers == 6:
        model = MLPClassifier(hidden_layer_sizes=(neurons, neurons), max_iter=maxIter, activation=activation, random_state=1)
    model.fit(feature_train, y_train_encoded)
    timestamp = int(time.time() * 1000)
    filename = f'MLP_Models/MLP_model_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}_{timestamp}.joblib'
    joblib.dump(model, filename)
    prediction = model.predict(feature_test)
    accuracy = accuracy_score(y_test_encoded, prediction)
    print(f'Accuracy MLP_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}: {accuracy:.2f}')
    inverse_pred = le.inverse_transform(prediction)
    inverse_test_labels = le.inverse_transform(y_test_encoded)
    cm = confusion_matrix(inverse_test_labels, inverse_pred, labels=le.inverse_transform(model.classes_))
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm.astype('int')
    print(f'Confusion Matrix MLP_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}:\n {cm}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(model.classes_))
    disp.plot()
    figname = f'MLP_Confusion_Matrix/4labels/Measurement3_4/MLP_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}.png'
    plt.savefig(figname)
    plt.show()

def train_test_SVM(data, kernel, C, dataset):
    if False:
        print('Hello World!')
    (feature_train, feature_test, label_train, label_test) = test_train_set(data)
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(feature_train, label_train)
    timestamp = int(time.time() * 1000)
    filename = f'SVM_Models/SVM_{kernel}_{C}_dataset{dataset}_{timestamp}.joblib'
    joblib.dump(model, filename)
    prediction = model.predict(feature_test)
    accuracy = accuracy_score(label_test, prediction)
    print(f'Accuracy SVM: {accuracy:.2f}')
    cm = confusion_matrix(label_test, prediction, labels=model.classes_)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm.astype('int')
    print(f'SVM_{kernel}_{C}_dataset{dataset}:\n {cm}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    figname = f'SVM_Confusion_Matrix/4labels/SVM_{kernel}_{C}_dataset{dataset}.png'
    plt.savefig(figname)
    plt.show()