from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import Utils


class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Instantiates the Model and compiles.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Predicts using the model on the data specified.
        """
        pass
