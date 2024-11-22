from sklearn.svm import SVC  
from model.base import BaseModel

class SVM(BaseModel):
    def __init__(self, kernel='linear', C=1.0) -> None:
        super(SVM, self).__init__()
        self.model = SVC(kernel=kernel, C=C)
        self.predictions = None
