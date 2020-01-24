import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.tests.test_sgd import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import pytest
from sklearn.preprocessing import LabelEncoder
import utils

DATASET_PATH = 'C:/Users/Delta/PycharmProjects/Adware-Detector/dataset/TotalFeatures-ISCXFlowMeter.csv'


def read_csv_file(file_path):
    return pd.read_csv(file_path)


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='micro')
    recall = recall_score(y_actual, y_predicted, average='micro')
    f1 = f1_score(y_actual, y_predicted, average='micro')

    return accuracy, precision, recall, f1


df = read_csv_file(DATASET_PATH)

y = df['class']
X_train = df.loc[:, df.columns != 'class']

y_train = y.replace({'benign': 0, 'asware': 1, 'GeneralMalware': 1})
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.70, test_size=0.30)

columns_name_list = list(df.columns.values)
utils.field_name_changer(df, columns_name_list)

fold = 0


# kfold = KFold(n_splits=5, random_state=42, shuffle=True)
#
# for train_index, test_index in kfold.split(X_train):
#     fold += 1
#     print("Fold: %s" % fold)
#
#     x_train_k, x_test_k = X_train.iloc[train_index], X_train.iloc[test_index]
#     y_train_k, y_test_k = y_train.iloc[train_index], y_train.iloc[test_index]
#
#     sgd_classifier = SGDClassifier(max_iter=1000, loss='hinge')
#     sgd_classifier.fit(x_train_k, y_train_k)
#
#     predictedValues = sgd_classifier.predict(x_test_k)
#
#     print("Accuracy SGDClassifier: %s"
#           % (accuracy_score(y_test_k, predictedValues)))
#     accuracy, precision, recall, f1 = calculate_metrics(y_test_k, predictedValues)
#     print("accuracy:" + str(accuracy))
#     print("precision:" + str(precision))
#     print("recall:" + str(recall))
#     print("f1:" + str(f1))
#
#     #############################################################
#     rand_forest_classifier = RandomForestClassifier(n_jobs=-1, max_depth=500)
#     rand_forest_classifier.fit(x_train_k, y_train_k)
#     predictedValues_rand_forest = rand_forest_classifier.predict(x_test_k)
#     print("Accuracy Random ForestClassifier: %s"
#           % (accuracy_score(y_test_k, predictedValues_rand_forest)))
#     accuracy, precision, recall, f1 = calculate_metrics(y_test_k, predictedValues_rand_forest)
#     print("accuracy:" + str(accuracy))
#     print("precision:" + str(precision))
#     print("recall:" + str(recall))
#     print("f1:" + str(f1))


def pca_decomposition(x_train):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_train)
    PCA_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    PCA_df = pd.concat([PCA_df, y], axis=1)
    PCA_df['calss'] = LabelEncoder().fit_transform(PCA_df['calss'])
    PCA_df.head()


def ica_decomposition(x_train):
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=3)
    X_ica = ica.fit_transform(x_train)


def lda_decomposition(x_train, y_train):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=1)

    # run an LDA and use it to transform the features
    X_lda = lda.fit(x_train, y_train).transform(x_train)
    print('Original number of features:', x_train.shape[1])
    print('Reduced number of features:', X_lda.shape[1])
