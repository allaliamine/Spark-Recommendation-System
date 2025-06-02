from jobs.Recommender import Recommender
from pyspark.sql import SparkSession

from flask import Flask, render_template, request

app = Flask(__name__)

def init():
    global recommender, spark, path_to_model, dataframe

    spark = SparkSession.builder \
        .appName("RecommenderApp") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    path_to_model = "/Users/mac/Desktop/Spark-Recommendation-System/models/custom_word2vec_model"
    recommender = Recommender(spark)

    dataframe = spark.read.parquet("/Users/mac/Desktop/Spark-Recommendation-System/data/final_data_with_embeddings.parquet")
    



@app.route("/")
def hello_world():
    init()
    return render_template("index.html")



@app.route("/ingredientRecommender", methods=["GET"])
def ingredientRecommender():
    
    return render_template("ingredientsRecommender.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_word = request.form["user_word"]

    df = recommender.recommend(dataframe, user_word, path_to_model, 10)
    df = df.select("url", "product_name", "brands", "nutriscore_grade", "nova_group", "image_url", "similarity")
    
    # Convert Spark DataFrame to a list of dictionaries for HTML rendering
    recommendations = df.toPandas().to_dict(orient='records')

    print(recommendations)
    
    return render_template("ingredientsRecommender.html", recommendations=recommendations, user_word=user_word)



