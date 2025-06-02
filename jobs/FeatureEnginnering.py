from jobs.UserDefinedFunctions import remove_hyphens_udf, explode_phrases_udf, remove_stopwords_udf
from pyspark.sql.functions import expr, array, flatten
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
import nltk

nltk.data.path.append('/Users/mac/Desktop/Spark-Recommendation-System/nltk_data')



class FeatureEngineering:
    def __init__(self, spark: SparkSession):
        self.spark = spark


    def addAllTokensColumn(self, df):

        df_with_tokens = df.withColumn(
            "categories_tokens",
            expr("""
                transform(
                    filter(split(categories_tags, ','), x -> startswith(x, 'en:')),
                    x -> replace(x, 'en:', '')
                )
            """)
        ).withColumn(
            "main_category_tokens",
            expr("""
                transform(
                    filter(split(main_category, ','), x -> startswith(x, 'en:')),
                    x -> replace(x, 'en:', '')
                )
            """)
        )

        df_with_tokens = df_with_tokens.withColumn(
            "all_tokens",
            flatten(array(
                "categories_tokens",
                "main_category_tokens",
            ))
        )

        df_with_tokens = df_with_tokens.withColumn("all_tokens", expr("array_distinct(all_tokens)"))
        return df_with_tokens
    

    def processTokens(self, df):
        df_with_tokens = self.addAllTokensColumn(df)

        df_with_tokens = df_with_tokens \
            .withColumn("all_tokens", remove_hyphens_udf("all_tokens")) \
            .withColumn("all_tokens", explode_phrases_udf("all_tokens")) \
            .withColumn("tokens", remove_stopwords_udf("all_tokens"))

        return df_with_tokens
    

    def addEmbedingColumn(self, df, path_to_model):
        word2vec_model = Word2VecModel.load(path_to_model)
        df_with_embeddings = word2vec_model.transform(df)
        return df_with_embeddings