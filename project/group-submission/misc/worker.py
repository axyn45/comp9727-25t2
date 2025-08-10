from multiprocessing import Pool
import pandas as pd
import spacy
import yake

lemmatizer = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor()

def f(x):
    return x*x

def get_bow(row:pd.Series, columns:list[str], lemmatizer):
    text = ' '.join([row[column] if not (pd.isna(row[column]) or row[column] is False) else '' for column in columns])
    doc = lemmatizer(text)
    lemmatized_text = " ".join(token.lemma_ for token in doc)
    keywords = kw_extractor.extract_keywords(lemmatized_text)
    bow = set(word for kw, _ in keywords for word in kw.split())
    return bow

def process_chunk(df_chunk:pd.DataFrame):
    return df_chunk.apply(
        get_bow,
        axis=1,
        columns=['title','selftext'],
        lemmatizer=lemmatizer,
    )

def parallel_bagging(df_chunks, num_processes=8):
    with Pool(num_processes) as pool:
        return pool.map(process_chunk, df_chunks)

def sample():
    p = Pool(2)
    with p:
        results = p.map(f, [1,2,3])
        return results

if __name__ == '__main__':
    print(sample())