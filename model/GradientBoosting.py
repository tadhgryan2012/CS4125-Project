from sklearn.ensemble import GradientBoostingClassifier
from model.base import BaseModel

class GradientBoosting(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3) -> None:
        super(GradientBoosting, self).__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.predictions = None
