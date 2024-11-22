from sklearn.neighbors import KNeighborsClassifier
from model.base import BaseModel

class KNN(BaseModel):
    def __init__(self, n_neighbors=5, metric='minkowski', p=2) -> None:
        super(KNN, self).__init__()
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p
        )
        self.predictions = None
