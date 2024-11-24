import os
import pickle

import pandas as pd
import numpy as np
import re

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer

from Config import *
from modelling.data_model import Data


class Util:
    @staticmethod
    def get_data_object(X: np.ndarray, df: pd.DataFrame, prediction: bool = False):
        return Data(X, df, prediction)

    @staticmethod
    def get_embeddings(df: pd.DataFrame):
        vectorizer_path = '../data/vectorizer.pkl'

        if Config.INTERACTION_CONTENT in df.columns:
            valid_indices = df[Config.INTERACTION_CONTENT].notnull()
            df = df[valid_indices]
        else:
            raise KeyError(f"Column {Config.INTERACTION_CONTENT} not found in dataframe.")

        if df.empty:
            raise ValueError("No valid rows in dataframe to process.")

        if os.path.exists('../data/vectorizer.pkl'):
            with open('../data/vectorizer.pkl', 'rb') as file:
                loaded_vectorizer = pickle.load(file)
            X = loaded_vectorizer.transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
        else:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
            with open('data/vectorizer.pkl', 'wb') as file:
                pickle.dump(vectorizer, file)

        print(f"Embeddings shape: {X.shape}, DataFrame shape: {df.shape}")
        return X, df

    @classmethod
    def load_data(cls, filename: str):
        data : pd.DataFrame
        filename_used = "AppGallery"
        if (filename == ""):
            print("loading original training data")
            data = cls.load_data_disk(filename_used, True)
        elif (os.path.exists(filename)):
            if (filename.endswith(".csv")):
                filename_used = re.split(r"[/, \ ]", filename)[-1].split(".")[0]
                data = cls.load_data_disk(filename)
            else:
                print("File that you provided is not a CSV file.")
                print("Loading original training data.")
                data = cls.load_data_disk(filename_used, True)
        else:
            print("File that you provided does not exist.")
            print("Loading original training data.")
            data = cls.load_data_disk(filename_used, True)

        return data, filename_used

    @staticmethod
    def load_processed_data(filename: str) -> DataFrame:
        filename = f"data/{filename}_processed.csv"
        data = pd.read_csv(filename)
        return data
    #
    @staticmethod
    def load_data_disk(filename: str, convert: bool = False) -> DataFrame:
        filename = f"data/{filename}.csv" if convert else filename
        data = pd.read_csv(filename)
        return data


