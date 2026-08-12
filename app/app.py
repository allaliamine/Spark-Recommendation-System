from flask import Flask, render_template, request
from jobs.Recommender import Recommender
from pyspark.sql import SparkSession
import google.generativeai as genai
from dotenv import load_dotenv
import os


app = Flask(__name__)

def init():
    global recommender, spark, path_to_model, dataframe, api_key

    load_dotenv()
    api_key = os.getenv("GEMINI-API-KEY")

    spark = SparkSession.builder \
        .appName("RecommenderApp") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    path_to_model = "/Users/amine/Desktop/Spark-Recommendation-System/models/custom_word2vec_model"
    recommender = Recommender(spark)

    dataframe = spark.read.parquet("/Users/amine/Desktop/Spark-Recommendation-System/data/final_data_with_embeddings.parquet")
    
def extract_ingredients(recipe_text):
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    prompt = f"Extract the ingredients from the following recipe as a comma-separated string with no instructions or quantities:\n\n{recipe_text}"

    response = model.generate_content(prompt)

    return response.text.strip()


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
    user_word = user_word.strip().lower()

    try:
        df = recommender.recommend(dataframe, user_word, path_to_model, 10)
    except ValueError as e:
        if "was not found in the vocabulary" in str(e):
            error_message = f"Sorry, '{user_word}' is not in our ingredient database. Please try a different ingredient."
            return render_template("ingredientsRecommender.html", error_message=error_message, user_word=user_word)
        else:
            raise e
    except Exception as e:
        error_message = "An unexpected error occurred. Please try again."
        return render_template("ingredientsRecommender.html", error_message=error_message, user_word=user_word)
        
    df = df.select("url", "product_name", "brands", "nutriscore_grade", "nova_group", "image_url", "similarity")
    
    recommendations = df.toPandas().to_dict(orient='records')
    
    return render_template("ingredientsRecommender.html", recommendations=recommendations, user_word=user_word, error_message=error_message if 'error_message' in locals() else None)


@app.route("/recipeRecommender", methods=["GET"])
def recipeRecommender():
    return render_template("recipeRecommender.html")


@app.route("/getIngredients", methods=["POST"])
def getIngredients():
    user_recipe = request.form["user_recipe"]
    ingredients_text = extract_ingredients(user_recipe)
    ingredients = list(ingredients_text.split(","))

    print(ingredients)
    
    return render_template("recipeRecommender.html",  ingredients=ingredients)
