import pandas as pd
import numpy as np
import graphviz

from sklearn import linear_model, preprocessing, cross_validation, metrics, tree

column_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

original_data = pd.read_csv('adult.data.csv', names = column_names, sep = r'\s*,\s*', engine = 'python', na_values = '?')

del original_data['fnlwgt']
del original_data["Education"]

binary_data = pd.get_dummies(original_data)
# Let's fix the Target as it will be converted to dummy vars too
binary_data["Target"] = binary_data["Target_>50K"]
del binary_data["Target_<=50K"]
del binary_data["Target_>50K"]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(binary_data[binary_data.columns.difference(["Target"])], binary_data["Target"], train_size=0.70)
scaler = preprocessing.StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = scaler.transform(X_test)

logrel = linear_model.LogisticRegression()

logrel.fit(X_train_scaled, y_train)
y_pred_logrel = logrel.predict(X_test_scaled)
print "Logistic regression F1 score: %f" % metrics.f1_score(y_test, y_pred_logrel)

dectree = tree.DecisionTreeClassifier(max_depth=4)
dectree = dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)
print "Decision tree F1 score: %f" % metrics.f1_score(y_test, y_pred)

dot_data = tree.export_graphviz(dectree, out_file='tree.gv', feature_names=binary_data.columns)
# graph = graphviz.Source(dot_data)
# graph.render('tree.gv')
