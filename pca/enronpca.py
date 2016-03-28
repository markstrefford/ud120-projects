import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

feature_1 = "bonus"
feature_2 = "long_term_incentive"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


def doPCA(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca

pca = doPCA(data)
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_data = pca.transform(data)
for ii, jj in zip(transformed_data, data):
    plt.scatter( first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
    plt.scatter( second_pc[0]*ii[1], second_pc[1]*ii[1], color="c")
    plt.scatter( jj[0], jj[1], color="b")

plt.xlabel("bonus")
plt.ylabel("long term incentive")
plt.savefig("enron_pca.pdf")
plt.show()
