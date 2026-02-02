

# Lab 6: Spark ML Pipeline on Amazon EMR (Customer Churn Prediction)

## Project Overview
[cite_start]This project implements an end-to-end distributed Machine Learning pipeline using Apache Spark on an Amazon EMR cluster[cite: 1, 7]. [cite_start]The goal is to predict bank customer churn based on the "Bank Customer Churn Dataset"[cite: 26, 27].

## Dataset Description
- [cite_start]**Source:** Kaggle Bank Customer Churn Dataset[cite: 28].
- [cite_start]**Target Variable:** `Exited` (0 = No churn, 1 = Churn)[cite: 39, 40].
- [cite_start]**Features:** Credit score, Geography, Gender, Age, Tenure, Balance, Number of products, and Estimated Salary [cite: 31-38].

## Steps Executed

### 1. Cluster Setup
- [cite_start]**Platform:** Amazon EMR (Release emr-7.12.0)[cite: 41].
- [cite_start]**Configuration:** 1 Primary (Master) node, 2 Core nodes [cite: 63-65].
- [cite_start]**Instance Type:** m4.large[cite: 49].
- [cite_start]**Applications:** Spark 3.5.6, Hadoop 3.4.1, Hive 3.1.3[cite: 61, 62].

### 2. Data Preparation and HDFS Upload
[cite_start]First, the dataset was uploaded to the Master node and then moved to HDFS for distributed processing [cite: 69-73]:

```bash
# Upload from local machine to EMR Master
scp -i rpc-key.pem Churn_Modelling.csv.xls hadoop@ec2-44-203-29-119.compute-1.amazonaws.com:/home/hadoop/

# On Master Node: Rename and Move to HDFS
mv Churn_Modelling.csv.xls Churn_Modelling.csv
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/

```

### 3. Environment Setup

To resolve dependencies for Spark ML, `numpy` was installed on the master node:

```bash
sudo pip install numpy

```

### 4. Running the ML Pipeline (Logistic Regression)

The main pipeline includes categorical indexing, one-hot encoding, vector assembly, and scaling .

**Execution Command:**

```bash
spark-submit --master yarn --deploy-mode client churn_pipeline.py

```

**Result:** Accuracy = **0.7929**

### 5. Experiment: Model Comparison (Option C)

A second script was created to compare Logistic Regression with a Random Forest Classifier.

**Execution Command:**

```bash
spark-submit --master yarn --deploy-mode client rf_pipeline.py

```

**Result:** Random Forest Accuracy = **0.8403**

## Conclusion

The Random Forest model performed significantly better (84.03%) than Logistic Regression (79.29%). This is due to the ensemble model's ability to capture non-linear patterns within the customer data that a linear model cannot easily identify.

## Monitoring

The execution was monitored using the **YARN Resource Manager UI** at port `8088`. The logs confirmed that tasks were distributed across 2 active core nodes.

