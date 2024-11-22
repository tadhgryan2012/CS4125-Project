from sklearn.naive_bayes import MultinomialNB
from model.base import BaseModel

class NaiveBayes(BaseModel):
    def __init__(self, alpha=1.0) -> None:
        super(NaiveBayes, self).__init__()
        self.model = MultinomialNB(alpha=alpha)
        self.predictions = None
