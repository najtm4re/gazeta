"""
Necessary libs for data processing
"""
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
import nltk
from tqdm import tqdm


class TenClosestNews:
    """
    Class for embeddings creation from gazeta dataset.
    """

    @staticmethod
    def delete_stopwords(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method deletes stopwords from summaries and creates
        new column named 'clean_summary'
        """
        nltk.download("stopwords")
        stopwords = set(nltk.corpus.stopwords.words("russian"))
        df["clean_summary"] = ""
        for idx in tqdm(df.index):
            words = df.loc[idx, "summary"].split()
            filtered_words = [
                word.lower() for word in words if word.lower() not in stopwords
            ]
            filtered_text = " ".join(filtered_words)
            df.loc[idx, "clean_summary"] = filtered_text
        return df

    def __init__(self):
        self.dataset = load_dataset("IlyaGusev/gazeta", revision="v2.0")
        df_list = []
        for name in self.dataset:
            df_list.append(self.dataset[name].to_pandas())
        self.df = pd.concat(df_list, ignore_index=True)

        self.encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.stopwords = set(nltk.corpus.stopwords.words("russian"))
        self.faiss_index = None

    def create_faiss_index(self):
        """
        This method creates embeddings from df_train of gazeta dataset.
        If there's pre-made embeddings .npy file, method just loads that one.
        These embeddings are used to create faiss index list.
        """
        current_file_path = os.path.abspath("bot_run.py")
        current_dir_path = os.path.dirname(current_file_path) + "/data"

        print(current_dir_path)

        if os.path.exists(f"{current_dir_path}/summary_embeddings.npy"):
            embeddings = np.load(f"{current_dir_path}/summary_embeddings.npy")
            print("=" * 79)
            print("Embeddings were loaded from a pre-made file")
            print("=" * 79)
        else:
            print("=" * 79)
            print(
                "Begin to create embeddings on the whole dataset. This can take a while..."
            )
            print("=" * 79)
            df_with_cleaned_summaries = TenClosestNews.delete_stopwords(self.df)
            embeddings = self.encoder.encode(
                df_with_cleaned_summaries.clean_summary, show_progress_bar=True
            )

        if not os.path.exists(f"{current_dir_path}\\summary_embeddings.npy"):
            np.save("summary_embeddings", embeddings)

        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        print("Faiss index has been succesfully created")
        print("=" * 79)

    def get_closest_news(self, query_text: str) -> pd.DataFrame:
        """
        This method creates list of top10 most relevant news and then
        sorting raw dataframe.
        """
        filtered_words = [
            word.lower()
            for word in query_text.split()
            if word.lower() not in self.stopwords
        ]
        filtered_query = " ".join(filtered_words)
        query_embedding = self.encoder.encode(filtered_query)

        _, sorted_index = self.faiss_index.search(np.array([query_embedding]), 10)

        response = self.df.iloc[sorted_index[0]]

        return response
