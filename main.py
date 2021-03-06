import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv('Datasets/Crop_recommendation.csv')
x_data = df[df.columns[:-1]]
y_data = df[df.columns[-1]]
print(y_data.value_counts())
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data, random_state=10)
KNN = KNeighborsClassifier(weights='distance')
KNN.fit(x_train, y_train)
yPredict = KNN.predict(x_test)
ac_knn = accuracy_score(y_test, yPredict) * 100
print('KNN Accuracy:', ac_knn)
SVM = SVC(kernel='rbf', gamma='scale',  cache_size=2000, decision_function_shape='ovo')
SVM.fit(x_train, y_train)
yPredict = SVM.predict(x_test)
ac_svm = accuracy_score(y_test, yPredict) * 100
print('SVM Accuracy', ac_knn)
begged = BaggingClassifier(n_estimators=30)
begged.fit(x_train, y_train)
yPredict = begged.predict(x_test)
ac_beg = accuracy_score(y_test, yPredict) * 100
print('Begged Tree Accuracy', ac_beg)

classifier = ['KNN', 'SVM', 'Begged Tree']
accuracy = [ac_knn, ac_svm, ac_beg]
plt.style.use('ggplot')
plt.barh(classifier, accuracy)
plt.title('Accuracies of Different Models')
plt.ylabel('Model')
plt.xlabel('Accuracy')
plt.show()
