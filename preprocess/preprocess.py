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

def preprocess_data(df):
    pipeline = PreprocessPipeline()
    pipeline.add_strategy(LoggingDecorator(DeduplicationStrategy()))
    pipeline.add_strategy(LoggingDecorator(NoiseRemovalStrategy()))
    #pipeline.add_strategy(LoggingDecorator(TranslationStrategy()))
    return pipeline.execute(df)