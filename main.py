
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Import classifiers
import get_images
import get_landmarks
import performance_plots
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm

# Load the data and their labels
image_directory = 'f22-dataset' # Use folder WithGlasses and WithoutGlasses for Experiments 1 & 2
X, y = get_images.get_images(image_directory)

# Get distances between face landmarks in the images
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 68, False)

# Comment this when using train/test split
# K-fold

# k = 5
# kf = KFold(n_splits=k, random_state=None)

#Comment the extra classifier when using a set of classifiers

# Matching and Decision - KNN - kfold
# acc_score = []
# model = ORC(knn())
# for train_index , test_index in kf.split(X):
#     X_train , X_test = X[train_index], X[test_index]
#     y_train , y_test = y[train_index] , y[test_index]
#     model.fit(X_train,y_train)
# matching_scores_knn = model.predict_proba(X_test)
# print(matching_scores_knn)

# Matching and Decision - SVM - Kfold
# acc_score = []
# model = ORC(svm(probability=True))
# for train_index , test_index in kf.split(X):
#     X_train , X_test = X[train_index], X[test_index]
#     y_train , y_test = y[train_index] , y[test_index]
#     model.fit(X_train,y_train)
# matching_scores_svm = model.predict_proba(X_test)
# print(matching_scores_svm)

# Matching and Decision - NB - Kfold
# acc_score = []
# model = ORC(GaussianNB())
# for train_index , test_index in kf.split(X):
#     X_train , X_test = X[train_index], X[test_index]
#     y_train , y_test = y[train_index] , y[test_index]
#     model.fit(X_train,y_train)
# matching_scores_NB = model.predict_proba(X_test)

# comment this when using k-fold

# train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Comment the extra classifier when using a set of classifiers

# Matching and Decision - GaussianNB - train/test split
# model = ORC(GaussianNB())
# model.fit(X_train, y_train)
# matching_scores_NB = model.predict_proba(X_test)

# Matching and Decision - KNN - train/test split
model = ORC(knn())
model.fit(X_train, y_train)
matching_scores_knn = model.predict_proba(X_test)

# Matching and Decision - SVM - train/test split
model = ORC(svm(probability=True))
model.fit(X_train, y_train)
matching_scores_svm = model.predict_proba(X_test)

# Fuse scores

# KNN & SVM
matching_scores = (matching_scores_knn + matching_scores_svm) / 2.0

# KNN & NB
# matching_scores = (matching_scores_NB + matching_scores_knn) / 2.0

# NB & SVM
# matching_scores = (matching_scores_svm + matching_scores_NB) / 2.0

gen_scores = []
imp_scores = []
classes = model.classes_
matching_scores = pd.DataFrame(matching_scores, columns=classes)

for i in range(len(y_test)):    
    scores = matching_scores.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])


# For classifiers kNN & SVM
performance_plots.performance(gen_scores, imp_scores, 'kNN-SVM-score_fusion', 100)

# For classifiers kNN & NB
# performance_plots.performance(gen_scores, imp_scores, 'kNN-NB-score_fusion', 100)

# For classifiers NB & SVM
# performance_plots.performance(gen_scores, imp_scores, 'NB-SVM-score_fusion', 100)



    
    
