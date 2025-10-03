import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv("E:/Folder Belajar Coding/Data analyst/Dataset/employee.csv")
x = df[['PaymentTier','Age','ExperienceInCurrentDomain']]
y = df['Education']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training, 30% test
clf = DecisionTreeClassifier() # Create a Decision Tree Classifier object
clf.fit(X_train, y_train) # Train the model
y_pred = clf.predict(X_test)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred_Nb = model.predict(X_test)
print(y_pred)
print("Predicting Employee Educations Based on Age, Experience, and Tier Payment using Decision Tree prediction, Accuracy of the model:", metrics.accuracy_score(y_test, y_pred))

print(y_pred_Nb)
print("Predicting Employee Educations Based on Age, Experience, and Tier Payment using Naive Bayes prediction, Accuracy of the model:", metrics.accuracy_score(y_test, y_pred_Nb))
plt.figure(figsize=(15,10))

tree.plot_tree(clf, feature_names=x.columns, class_names=y)
plt.show()