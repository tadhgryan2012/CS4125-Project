import pandas as pd
from googletrans import Translator
import re


class DeduplicationStrategy:
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Removing duplicate entries...")
        print(f"No. of rows before: {len(df)}")
        df = df.drop_duplicates(["Ticket id", "Interaction id"])
        print(f"No. after: {len(df)}")
        return df

class NoiseRemovalStrategy:
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Removing noise from entries...")
        noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"

        for col in ["Interaction content", "Ticket Summary"]:
            df.loc[:, col] = df[col].fillna("").str.lower().str.strip()
            df.loc[:, col] = df[col].replace(noise, " ", regex=True).replace(r"\s+", " ", regex=True).str.strip()
        return df


class TranslationStrategy:
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Translating entries to English...")
        df["Ticket Summary"] = df["Ticket Summary"].apply(lambda x: f"Translate({x})" if isinstance(x, str) else x)
        return df