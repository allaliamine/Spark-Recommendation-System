from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.ml.feature import Word2VecModel


class Recommender:

    def __init__(self, spark: SparkSession):
        self.spark = spark
        

    def getUserInputAsVector(self, user_word, loaded_model):
        try:
            user_vec_row = loaded_model.getVectors().filter(f"word = '{user_word}'").collect()[0]           
            user_vec_np = np.array(user_vec_row['vector'])
        except IndexError:
            raise ValueError(f"The word '{user_word}' was not found in the vocabulary.")
        
        return user_vec_np
    
    def getCosineSimilarity(self, df, user_vec_np):

        user_vec_broadcast = self.spark.sparkContext.broadcast(user_vec_np)

        def cosine_similarity(vec):
            vec = np.array(vec)
            user_vec = user_vec_broadcast.value
            dot = np.dot(vec, user_vec)
            norm1 = np.linalg.norm(vec)
            norm2 = np.linalg.norm(user_vec)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))

        cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        df_with_similarity = df.withColumn("similarity", cosine_similarity_udf(col("embedding")))

        return df_with_similarity
    

    def recommend(self, df, user_word, path_to_model ,limit):
        word2vec_model = Word2VecModel.load(path_to_model)
        user_word_vec = self.getUserInputAsVector(user_word, word2vec_model)
        df_with_similarity = self.getCosineSimilarity(df, user_word_vec)

        return df_with_similarity.orderBy(col("similarity").desc()).limit(limit)