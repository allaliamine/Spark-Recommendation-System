from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import col, when, lit, round, split, explode, trim
from functools import reduce  
import operator


class Cleaner:

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # country should be in the a format such as this : "fr" for France, "en" for England, etc.
    def filterCountry(self, df, country: str):
        filtered_df = df.filter(col("countries_tags").like(f"%en:{country}%"))

        return filtered_df
    
    def filterAllNullColumns(self, df):

        # the columns were chosen based on the script run on the cleaning notebook
        null_columns = ['cities', 'allergens_en', 'additives', 'nutrition-score-uk_100g', 'carbohydrates-total_100g']
        cleaned_df = df.drop(*null_columns)        
        return cleaned_df
    

    # this function is used to filter out the metadata columns that are not needed for the recommender system
    # some columns are needed for just filter we may keep them after testing 
    # the main object now is to get a clean dataframe with only columns that are needed for the recommender system
    def filterMetaDataColumns(self, df):
        columns_to_drop = [col for col in df.columns if 
                   "origins" in col or 
                   "manufacturing_places" in col or
                   "cities" in col or
                   "countries" in col or
                   "owner" in col or
                   "packaging" in col or 
                   "emb_codes" in col or 
                   "countries" in col or 
                   "states" in col or 
                   "100g" in col]

        cleaned_df = df.drop(*columns_to_drop)
        
        return cleaned_df


    # this function is used to filter out the columns that are not useful for the recommender system
    def filterNonUsefullColumns(self, df):
        # the columns were chosen based openfoodfacts columns description
        non_usefull_columns = [ "creator", "last_modified_by", "abbreviated_product_name", "categories",
                       "labels", "purchase_places", "stores", "ingredients_text", "allergens",
                        "traces", "traces_en", "serving_size", "serving_quantity", "no_nutrition_data", "additives_n", "additives_en",
                        "nutriscore_score", "pnns_groups_1", "pnns_groups_2", "food_groups", "food_groups_en",
                        "environmental_score_grade", "nutrient_levels_tags", "data_quality_errors_tags", "unique_scans_n", "main_category_en",
                        "image_small_url","image_ingredients_url", "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
                        "popularity_tags", "brands_tags", "brands_en", "traces_tags","food_groups_tags","completeness"]
        cleaned_df = df.drop(*non_usefull_columns)
        
        return cleaned_df
        

    # we will use this function to calculate the completeness of each row (product)
    def calculateCompletenessOfNeededCOlumns(self, df):
        columns_to_calculate_completeness = ['product_name', 'generic_name', 'categories_en', 
                                     'ingredients_tags', 'ingredients_analysis_tags', 'main_category']

        non_null_exprs = [
            when((col(c).isNotNull()) | (col(c) != ""), 1).otherwise(0) for c in columns_to_calculate_completeness
        ]

        non_null_sum = reduce(operator.add, non_null_exprs)


        df_with_completeness = df.withColumn(
            "completeness",
            round(non_null_sum / lit(len(columns_to_calculate_completeness)), 3)
        )
        
        return df_with_completeness
    

    def filterNonCopmpleteProducts(self, df, threshold: float = 0.5):

        cleaned_df = df.filter(col("completeness") > threshold)
        
        return cleaned_df
    
    # we will drop the rows that have null values in the product_name, categories_tags, main_category
    def cleanNullRows(self, df):

        cleaned_df = df.na.drop(subset=["product_name", "categories_tags", "main_category"])
        
        return cleaned_df
    


    # this function is used to filter out the main columns that are not in English
    # main_category, categories_tags
    def filterNonEnglishColumns(self, df):

        split_main = df \
            .withColumn("main_split", split(col("main_category"), ",")) \
            .withColumn("main_exploded", explode("main_split")) \
            .withColumn("main_exploded", trim(col("main_exploded")))

        split_categories = df \
            .withColumn("cat_split", split(col("categories_tags"), ",")) \
            .withColumn("cat_exploded", explode("cat_split")) \
            .withColumn("cat_exploded", trim(col("cat_exploded")))

        non_en_main_codes = split_main.filter(~col("main_exploded").startswith("en:")).select("code").distinct()
        non_en_cat_codes = split_categories.filter(~col("cat_exploded").startswith("en:")).select("code").distinct()

        non_en_codes = non_en_main_codes.union(non_en_cat_codes).distinct()

        english_only_df = df.join(non_en_codes, on="code", how="left_anti")

        return english_only_df
