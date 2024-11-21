#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
# from modelling.modelling import *
from modelling.ModelFactory import ModelFactory
from modelling.data_model import *
import random
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 0
random.seed(seed)
np.random.seed(seed)

def de_duplication(df: pd.DataFrame):
    """Remove duplicate rows."""
    print("Removing duplicate rows...")
    df = df.drop_duplicates()
    return df

def noise_remover(df: pd.DataFrame):
    """Clean noise from all text columns."""
    print("Removing noise from all columns...")
    for col in df.select_dtypes(include='object'):  # Only process string columns
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True).str.strip()
    return df

def translate_to_en(text_list: list):
    """Translate a list of text strings to English."""
    print("Translating text list to English...")
    # TODO - Add access to translate API to actually translate text
    translated_list = [f"Translated({text})" if isinstance(text, str) else text for text in text_list]
    return translated_list

def get_tfidf_embd(df: pd.DataFrame):
    """Generate TF-IDF embeddings for all text columns."""
    print("Generating TF-IDF embeddings for all text columns...")
    vectorizer = TfidfVectorizer()
    tfidf_matrices = {}
    for col in df.select_dtypes(include='object'):  # Only process string columns
        tfidf_matrix = vectorizer.fit_transform(df[col].fillna(''))
        tfidf_matrices[col] = tfidf_matrix
    return tfidf_matrices

def load_data():
    print("get_input_data()")
    data = pd.read_csv('data/AppGallery.csv')
    return data

def preprocess_data(df: pd.DataFrame):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    vectorizer = TfidfVectorizer()

    # Generate embeddings only for non-null rows
    valid_indices = df[Config.INTERACTION_CONTENT].notnull()
    df = df[valid_indices]
    X = vectorizer.fit_transform(df[Config.INTERACTION_CONTENT].fillna('')).toarray()
    
    print(f"Embeddings shape: {X.shape}, DataFrame shape: {df.shape}")
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)



def perform_modelling(name: int, data: Data):
    """
    Perform model prediction, evaluation, and optionally save predictions.
    """
    modelFactory = ModelFactory()
    modelFactory.create_model(name)
    modelFactory.train_evaluate(data)
    # modelFactory.predict(data)

# This is for saving predictions into predictions.csv - can help create reports w model perf.
def save_predictions(df, predictions, output_path="predictions.csv"):
    """
    Save predictions to a CSV file.
    """
    # Use the test DataFrame to align predictions
    test_df = df.iloc[df.index.isin(data.get_test_df().index)]
    test_df['Predictions'] = predictions
    
    # Save to CSV
    test_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}.")


# Code will start executing from following line
if __name__ == '__main__':
    # pre-processing steps
    df = load_data()
    
    df = preprocess_data(df)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(df)

    # data modelling
    data = get_data_object(X, df)
    
    # modelling
    perform_modelling(0b001, data)
