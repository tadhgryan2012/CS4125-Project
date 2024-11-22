from sklearn.linear_model import LogisticRegression
from model.base import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.model = LogisticRegression(max_iter=1000)
