from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

forest = RandomForestClassifier(n_estimators=1, random_state=0) #n_estimators=1, random_state=0
forest.fit(X_train, y_train)

#Correction
print("Train set correction: {:.3f}".format(forest.score(X_train, y_train)))
print("Test set correction: {:.3f}".format(forest.score(X_test, y_test)))

#-----------------------------------
'''import mglearn
import matplotlib.pyplot as plt

#Visualization
fig, axes = plt.subplots(2, 3, figsize=(20,10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree{}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)'''
