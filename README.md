# Data Mining Project

## Project Structure

```
dmproject/
│
├── resources/
│   ├── original_data/         # Raw CSV data files
│   ├── processed_data/        # Processed ARFF files and results
│   ├── model/                 # Saved models from normal training
│   └── model_10Fold/          # Saved models from 10-fold cross-validation
│
├── src/
│   └── dmproject/
│       └── Main.java          # Main Java pipeline for data processing and modeling
│
├── draw.ipynb                 # Jupyter Notebook for data/result visualization
└── README.md                  # This guide
```

---

## Pipeline Steps and Their Purpose

### **1. Data Conversion**
- **Function:** `convertAllCsvToArff()`
- **Purpose:** Converts raw CSV files to ARFF format for Weka processing.
- **How to use:** This is the first step in `Main.java`. It prepares your data for all subsequent steps.

### **2. Data Integration & Feature Engineering**
- **Functions:** `processFiles()`, `processFileCombine()`
- **Purpose:** Integrates multiple data sources and creates new features (statistical aggregations, etc.) for modeling.
- **How to use:** These steps are called after data conversion and before any modeling.

### **3. Data Preprocessing**
- **Functions:**  
  - `preprocessTrainDataClean`, `preprocessTestDataClean`  
  - `preprocessTrainDataCluster`, `preprocessTestDataCluster`  
  - `preprocessTrainDataGroup`, `preprocessTestDataGroup`  
  - ...and variants with outlier removal and normalization
- **Purpose:** Cleans data (removes missing values, duplicates, outliers), normalizes, and optionally clusters or groups data for more robust modeling.
- **How to use:** These are called on both training and test sets before model training.

### **4. Model Training, Evaluation & Testing**
- **Function:** `buildTestModel(...)`, `buildModel(...)`, `testModel(...)`
- **Purpose:**  
  - `buildTestModel(...)`: This is the main step for both training and evaluating models. It trains the model on your processed data, saves the trained model to disk, and immediately evaluates it on test/sample datasets, printing metrics and saving results.
  - `buildModel(...)`: Only trains and saves the model (no evaluation).
  - `testModel(...)`: Only loads a previously saved model and evaluates it on test/sample data.
- **How to use:**  
  - Use `buildTestModel(...)` after preprocessing to train, save, and evaluate your models in one step.
  - Use `buildModel(...)` if you only want to train and save the model for later use.
  - Use `testModel(...)` if you want to evaluate a model that was trained previously (for example, on new data or after cross-validation).

#### **Example: Evaluating with 10-fold trained models**
You can also evaluate models that were trained using 10-fold cross-validation by calling `testModel` with the appropriate model and data paths, for example:
```java
// Evaluate using 10-fold trained RandomForest model
testModel("resources/model_10Fold/RandomForest_data_clean_model_10Fold.model",
          "resources/processed_data/data_test_clean.arff",
          "resources/processed_data/sample_clean.arff",
          "data clean");

// Evaluate using 10-fold trained MultilayerPerceptron model (with group & outlier removal)
testModel("resources/model_10Fold/MultilayerPerceptron_data_Group_Out_clean_model_10Fold.model",
          "resources/processed_data/data_test_Group_Out_clean.arff",
          "resources/processed_data/sample_Group_Out_clean.arff",
          "data group clean remove outline");
```
You can change these paths to use different models or datasets as needed.

---

### **5. Cross-Validation for Model Comparison**
- **Function(s) used:**  
  - `runModelFold("resources/processed_data/data_train_clean.arff");`
  - `runModelFold("resources/processed_data/data_train_clean_Cluster.arff");`
  - `runModelFold("resources/processed_data/data_train_Group_clean.arff");`
  - `runModelFold("resources/processed_data/data_train_Group_Out_clean.arff");`
  - `runModelFold("resources/processed_data/data_train_Group_Out_Cluster_clean.arff");`
- **Purpose:**  
  - Performs 10-fold cross-validation on each processed training dataset.
  - Compares the performance of different preprocessing strategies and models in a robust, unbiased way.
  - Outputs evaluation metrics (e.g., RMSE, MAE, correlation) for each model and dataset combination.
- **How to use:**  
  - Call `runModelFold` for each ARFF file you want to evaluate with cross-validation.
  - Results are printed to the console for comparison.

---

### **6. Evaluation Using 10-Fold Trained Models**
- **Function(s) used:**  
  - `testModel("resources/model_10Fold/RandomForest_data_clean_model_10Fold.model", "resources/processed_data/data_test_clean.arff", "resources/processed_data/sample_clean.arff", "data clean");`
  - `testModel("resources/model_10Fold/MultilayerPerceptron_data_Group_Out_clean_model_10Fold.model", "resources/processed_data/data_test_Group_Out_clean.arff", "resources/processed_data/sample_Group_Out_clean.arff", "data group clean remove outline");`
- **Purpose:**  
  - Loads models that were trained using 10-fold cross-validation.
  - Evaluates these models on the corresponding test and sample datasets.
  - Provides final validation metrics to see how well the cross-validated models generalize to unseen data.
- **How to use:**  
  - Call `testModel` with the path to the 10-fold trained model and the appropriate test/sample ARFF files.
  - Results are printed to the console for analysis.

---

## Customizing Input/Output Paths

You can **change the file paths** in the Java code to use different input data or to save models/results to other locations.  
For example, you can modify the arguments to `buildTestModel`, `buildModel`, `testModel`, or `runModelFold` to point to your own ARFF files or to save/load models from a different directory.

---

## Visualization with `draw.ipynb`

- **Purpose:**  
  `draw.ipynb` is a Jupyter Notebook for visualizing your data and model results. You can use it to:
  - Plot feature distributions, correlations, and outlier detection
  - Visualize model predictions vs. ground truth
  - Compare model performances with charts (bar, boxplot, etc.)

- **How to use:**
  1. Make sure you have Python 3 and Jupyter installed (`pip install jupyter pandas matplotlib seaborn scipy scikit-learn liac-arff`).
  2. Place `draw.ipynb` in the `dmproject` folder.
  3. Run `jupyter notebook draw.ipynb` from your terminal.
  4. Follow the notebook instructions to load ARFF files or result files and generate visualizations.

---

## Summary Table

| Step                | Java Function(s)                  | Purpose                                      | Output/Usage                |
|---------------------|-----------------------------------|----------------------------------------------|-----------------------------|
| Data Conversion     | `convertAllCsvToArff`             | CSV → ARFF                                   | ARFF files                  |
| Data Integration    | `processFiles`, `processFileCombine` | Merge, feature engineering                | Combined ARFF files         |
| Preprocessing       | `preprocess*`                     | Clean, normalize, cluster/group, outlier     | Cleaned ARFF files          |
| Model Training      | `buildModel`                      | Train and save models                        | Model files (`model/`)      |
| Model Evaluation & Testing | `buildTestModel`, `testModel` | Train, evaluate, and test models           | Metrics, results, models    |
| Cross-Validation    | `runModelFold`                    | 10-fold cross-validation                     | Metrics, results, models in `model_10Fold/` |
| 10-Fold Model Evaluation | `testModel`                  | Evaluate 10-fold trained models on test/sample | Metrics, results            |
| Visualization       | `draw.ipynb`                      | Data/result visualization (Python notebook)   | Plots, charts               |

---

## Dataset Download

- **Dataset using in this project:**  
https://www.kaggle.com/datasets/podsyp/production-quality

---

## About

- **Project:** Data Mining for Coffee Roasting Quality Prediction
- **Authors:** Huỳnh Hữu Nghĩa
- **Access to Datasets and Trained Models:**

---

**Tip:**  
You can comment/uncomment steps in `Main.java` to run only the parts you want, making it easy to debug or experiment with each stage of the pipeline.