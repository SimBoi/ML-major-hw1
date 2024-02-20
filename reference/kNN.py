from scipy.spatial import distance
from sklearn.base import BaseEstimator,ClassifierMixin
def get_label(label):
  return np.sign(np.sum(label))

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
      self.n_neighbors = n_neighbors
      self.data=None
      self.labels=None
      
    def fit(self, X, y):
      self.data = np.copy(X)
      self.labels = np.copy(y)
      return self

    def predict(self, X):
      # Note: You can use self.n_neighbors here
      predictions = None
      if self.data is not None and self.labels is not None:
        dist= distance.cdist(X,self.data ,'euclidean')
        k_near=np.argpartition(dist, self.n_neighbors,axis=1)[:,:self.n_neighbors]
        k_near_labels=self.labels[k_near]
        predictions=np.apply_along_axis(get_label,axis=1,arr=k_near_labels)
       
      return predictions

      