from pyspark.sql import SparkSession


class ReadSaveJob:

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read_csv_data(self, input_path :str , sep):
        df = self.spark.read.csv(input_path, sep=sep ,header=True)
        print(f"Data read from {input_path} with separator '{sep}'")
        print(f"Schema of the DataFrame:\n{df.printSchema()}")
        return df
    

    def read_parquet_data(self, input_path: str):
        df = self.spark.read.parquet(input_path)
        print(f"Schema of the DataFrame:\n{df.printSchema()}")
        return df


    def save_data(df, output_path: str):
        df.write.option("header", True).mode("overwrite").csv(output_path)

