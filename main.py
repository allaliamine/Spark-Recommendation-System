from jobs.ReadSave import ReadSaveJob as ReadSave
from jobs.Recommender import Recommender
from jobs.Cleaner import Cleaner
from jobs.FeatureEnginnering import FeatureEngineering


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spark Recommendation System") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()


readSave = ReadSave(spark)
cleaner = Cleaner(spark)
fetaure = FeatureEngineering(spark)

# Word2Vec model
path_to_model = "/Users/mac/Desktop/Spark-Recommendation-System/models/custom_word2vec_model"

# read data from csv file
df = readSave.read_csv_data("/Users/mac/Desktop/Spark-Recommendation-System/data/en.openfoodfacts.org.products.csv.gz")
recommender = Recommender(spark)

# clean data
df_country = cleaner.filterCountry(df, "fr")
df_cleaned = cleaner.filterAllNullColumns(df_country)
df_cleaned = cleaner.filterMetaDataColumns(df_cleaned)
df_cleaned = cleaner.filterNonUsefullColumns(df_cleaned)
df_cleaned = cleaner.calculateCompletenessOfNeededCOlumns(df_cleaned)
df_cleaned = cleaner.filterNonCopmpleteProducts(df_cleaned, 0.5)
df_cleaned = cleaner.cleanNullRows(df_cleaned)
df_cleaned = cleaner.filterNonEnglishColumns(df_cleaned)

# feature engineering
df_with_tokens = fetaure.addAllTokensColumn(df_cleaned)
df_with_tokens = fetaure.processTokens(df_with_tokens)
df_with_embeddings = fetaure.addEmbedingColumn(df_with_tokens, "/Users/mac/Desktop/Spark-Recommendation-System/models/custom_word2vec_model")

# save the final dataframe with embeddings
df_with_embeddings.write.mode("overwrite").parquet("/Users/mac/Desktop/Spark-Recommendation-System/data/final_data_with_embeddings.parquet")

# read the final dataframe with embeddings
dataframe = spark.read.parquet("/Users/mac/Desktop/Spark-Recommendation-System/data/final_data_with_embeddings.parquet")

# get on user input
user_word = input("Enter an ingredient to get recommendations: ")

# get recommendations based on user input
recommendations = recommender.recommend(df, user_word, path_to_model, limit=10)

# show recommendations
recommendations.show(truncate=False)
