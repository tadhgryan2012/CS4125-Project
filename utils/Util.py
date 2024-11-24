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
        vectorizer_path = 'data/vectorizer.pkl'

        if Config.INTERACTION_CONTENT in df.columns:
            valid_indices = df[Config.INTERACTION_CONTENT].notnull()
            df = df[valid_indices]
        else:
            raise KeyError(f"Column {Config.INTERACTION_CONTENT} not found in dataframe.")

        if df.empty:
            raise ValueError("No valid rows in dataframe to process.")

        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as file:
                loaded_vectorizer = pickle.load(file)
            X = loaded_vectorizer.transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
        else:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
            with open(vectorizer_path, 'wb') as file:
                pickle.dump(vectorizer, file)

        print(f"Embeddings shape: {X.shape}, DataFrame shape: {df.shape}")
        return X, df

    @staticmethod
    def get_data(filename: str = "", default_path: str = 'data/AppGallery.csv'):
        if (filename):
            if os.path.exists(filename) and filename.endswith(".csv"):
                print(f"Using data: {filename}")
                return pd.read_csv(filename)
            else: FileNotFoundError(f"The file '{filename}' does not exist or is not a valid .csv file.")
        else:
            if os.path.exists(default_path.replace('.csv', '_processed.csv')):
                print(f"Using data: {default_path.replace('.csv', '_processed.csv')}")
                return pd.read_csv(default_path.replace('.csv', '_processed.csv'))
            else:
                print(f"Using data: {default_path}")
                return pd.read_csv(default_path)
