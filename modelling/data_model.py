import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
                 # This method will create the model for data
                 #This will be performed in second activity
        print("Data.__init__()")

        # Validate alignment
        if len(X) != len(df):
            raise ValueError(f"Embeddings (X) and DataFrame (df) lengths do not match: {len(X)} vs {len(df)}")

        self.embeddings = X
        self.df = df
        
        #Train/test split. - IDK if we need train_df & test_df, I added because was included in with init commit.
        self.X_train, self.X_test, self.y_train, self.y_test, train_indices, test_indices = train_test_split(
            X, df[Config.CLASS_COL].fillna(''), df.index, test_size=0.2, random_state=seed
        )

        # Maintain traceability
        self.train_df = self.df.iloc[train_indices]
        self.test_df = self.df.iloc[test_indices]

  
    def get_embeddings(self):
        return self.embeddings
    def get_X_train(self):
        return self.X_train
    def get_X_test(self):
        return self.X_test
    def get_type_y_train(self):
        return self.y_train
    def get_type_y_test(self):
        return self.y_test
    def get_train_df(self):
        return self.train_df
    def get_test_df(self):
        return self.test_df
    def get_type(self):
        return self.df[Config.CLASS_COL].values

