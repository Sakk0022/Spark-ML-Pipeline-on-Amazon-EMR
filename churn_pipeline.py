from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Инициализация сессии Spark
spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()

# 2. Загрузка данных из HDFS [cite: 109, 110]
data = spark.read.csv("hdfs:///user/hadoop/churn_input/Churn_Modelling.csv", header=True, inferSchema=True)

# 3. Индексация категориальных колонок [cite: 114, 115]
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

# 4. Кодирование (OneHotEncoder) [cite: 118, 119]
encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# 5. Сборка признаков в вектор (VectorAssembler) [cite: 120-128]
assembler = VectorAssembler(
    inputCols=[
        "CreditScore", "Age", "Tenure", "Balance", 
        "NumOfProducts", "EstimatedSalary", "GeographyVec", "GenderVec"
    ],
    outputCol="features"
)

# 6. Масштабирование признаков [cite: 129-132]
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# 7. Настройка модели (Логистическая регрессия) [cite: 133-136]
lr = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures")

# 8. Создание Pipeline [cite: 137-145]
pipeline = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler, scaler, lr])

# 9. Обучение модели [cite: 147, 148]
model = pipeline.fit(data)

# 10. Предсказание [cite: 149, 150]
predictions = model.transform(data)
predictions.select("Exited", "prediction", "probability").show(10)

# 11. Оценка точности [cite: 153-161]
evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Total Accuracy: {accuracy}")

spark.stop()