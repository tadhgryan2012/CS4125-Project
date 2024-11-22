import stanza
import pandas as pd
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from Config import *

class DeduplicationStrategy:
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Removing duplicate entries...")
        df.drop_duplicates(inplace=True)
        print(f"No. of rows before: {len(df)} and after: {len(df)}")
        return df

class NoiseRemovalStrategy:
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Removing noise from entries...")
        noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"

        for col in ["Interaction content", "Ticket Summary"]:
            df[col] = df[col].fillna("").str.lower().str.strip()
            df[col] = df[col].replace(noise, " ", regex=True)
            df[col] = df[col].replace(r"\s+", " ", regex=True)
        return df


class TranslationStrategy:
    def __init__(self, column=Config.INTERACTION_CONTENT):
        self.column = column


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Translating entries to English...")
        df["translation for " + self.column] = self.translate(df)
        return df

    def translate(self, df: pd.DataFrame) -> pd.DataFrame:
        t2t_m = "facebook/m2m100_418M"
        t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

        model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
        tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
        nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                     download_method=DownloadMethod.REUSE_RESOURCES)

        text_en_l = []
        for text in df[self.column]:
            if text == "":
                text_en_l = text_en_l + [text]
                continue

            doc = nlp_stanza(text)
            print(doc.lang)
            if doc.lang == "en":
                text_en_l = text_en_l + [text]
            else:
                lang = doc.lang
                if lang == "fro":  # fro = Old French
                    lang = "fr"
                elif lang == "la":  # latin
                    lang = "it"
                elif lang == "nn":  # Norwegian (Nynorsk)
                    lang = "no"
                elif lang == "kmr":  # Kurmanji
                    lang = "tr"
                elif lang == "mt":
                    lang = "pl"

                case = 2

                if case == 1:
                    text_en = t2t_pipe(text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                    text_en = text_en[0]['generated_text']
                elif case == 2:
                    tokenizer.src_lang = lang
                    encoded_hi = tokenizer(text, return_tensors="pt")
                    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                    text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    text_en = text_en[0]
                else:
                    text_en = text

                text_en_l = text_en_l + [text_en]

                print(text)
                print(text_en)

        return text_en_l