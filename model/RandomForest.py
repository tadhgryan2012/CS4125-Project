import numpy as np
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        seed = 0
        np.random.seed(seed)
        self.model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
