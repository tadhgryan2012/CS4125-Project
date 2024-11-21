class PreprocessingDecorator:
    def __init__(self, base_strategy):
        self.base_strategy = base_strategy

    def execute(self, df):
        return self.base_strategy.execute(df)

class LoggingDecorator(PreprocessingDecorator):
    def execute(self, df):
        print("Starting preprocessing step...")
        df = self.base_strategy.execute(df)
        print("Preprocessing step completed")
        return df