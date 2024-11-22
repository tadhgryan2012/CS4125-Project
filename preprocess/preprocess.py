# Methods related to data loading and all pre-processing steps will go here
from .strategies import DeduplicationStrategy, NoiseRemovalStrategy, TranslationStrategy
from .decorators import LoggingDecorator
from utils.interfaces import PreprocessStrategy

class PreprocessPipeline:
    def __init__(self):
        self.strategies = []

    def add_strategy(self, strategy: PreprocessStrategy):
        self.strategies.append(strategy)

    def execute(self, df):
        for strat in self.strategies:
            df = strat.execute(df)
        return df

def preprocess_data(df, strategy):
    pipeline = PreprocessPipeline()

    for strat in strategy:
        pipeline.add_strategy(LoggingDecorator(strat))

    processed_data = pipeline.execute(df)

    output_file = "data/AppGallery_processed.csv"
    processed_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

    return processed_data