from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, DoubleType
from nltk.corpus import stopwords
import numpy as np

stop_words = set(stopwords.words('english'))


def remove_hyphens(tokens):
    return [token.replace("-", " ") for token in tokens]


def explode_phrases(tokens):
    words = []
    for token in tokens:
        words.extend(token.lower().split())
    return list(set(words))


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]



def cosine_similarity(vec, user_vec_broadcast):
    
    vec = np.array(vec)
    user_vec = user_vec_broadcast.value
    dot = np.dot(vec, user_vec) 
    norm1 = np.linalg.norm(vec)
    norm2 = np.linalg.norm(user_vec)
    if norm1 == 0 or norm2 == 0:
         return 0.0
    return float(dot / (norm1 * norm2))



remove_hyphens_udf = udf(remove_hyphens, ArrayType(StringType()))
explode_phrases_udf = udf(explode_phrases, ArrayType(StringType()))
remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
cosine_similarity_udf = udf(cosine_similarity, DoubleType())