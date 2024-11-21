from abc import ABC, abstractmethod

import pandas as pd


class PreprocessStrategy(ABC):
    @abstractmethod
    def execute(self,df: pd.DataFrame) -> pd.DataFrame:
        pass