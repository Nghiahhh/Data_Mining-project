# Data Mining Project

## Project Structure

```
Data_Mining-project/
│
├── resources/
│   ├── original_data/         # Raw CSV data files
│   ├── processed_data/        # Processed ARFF files and results
│   ├── model/                 # Saved models from normal training
│   └── model_10Fold/          # Saved models from 10-fold cross-validation
│
├── src/
│   └── data_mining/
│       └── project/
│           └── Data_MiningProject.java   # Main Java pipeline for data processing and modeling
│
├── draw.ipynb                 # Jupyter Notebook for data/result visualization
└── README.md                  # This guide
```

---

## Pipeline Steps (Matching the Code Structure)

### 1. Input File Paths & Convert CSV to ARFF
- **Function:** `convertAllCsvToArff(pathX, pathY, pathSubmission)`
- **Purpose:** Convert raw CSV files to ARFF format for Weka processing.
- **How to use:** The program prompts for file paths at runtime. This is the first step in `main`.

### 2. Data Integration & Feature Engineering
- **Functions:** `processFiles(pathX, pathY, pathSubmission)`, `processFileCombine(pathX, pathY, pathSubmission)`
- **Purpose:** Integrate multiple data sources and create new features (statistical aggregations, etc.) for modeling.
- **How to use:** Called immediately after data conversion.

### 3. Data Preprocessing, Splitting, and Saving
- **Steps:**
  - Load ARFF files created in previous steps.
  - Shuffle data (`randomize`).
  - Split data into train/test (70/30).
  - Preprocess data:
    - `preprocessTrainDataClean`, `preprocessTestDataClean`
    - `preprocessTrainDataCleanCluster`, `preprocessTestDataCleanCluster`
    - `preprocessTrainDataGroupRemoveOutliers`, `preprocessTestDataGroupRemoveOutliers`
  - Save processed ARFF files (clean, cluster, group, combine).

### 4. Feature Aggregation (Minute-level to Feature-level)
- **Function:** `aggregateMinuteFeaturesAndSave(inputPath, outputPath)`
- **Purpose:** Aggregate minute-level data into statistical features (mean, max, min, delta, etc.).
- **How to use:** Called for train/test/sample sets.

### 5. Model Training, Evaluation, and Testing
- **Functions:**  
  - `buildTestModel(trainPath, testPath, file)`
  - `buildTestModelCluster(trainPath, testPath, file)`
  - `buildModel(trainPath, file)`
  - `buildModelCluster(trainPath, file)`
  - `testModel(modelPath, testPath, file)`
  - `testModelCluster(modelPath, testPath, file)`
- **Purpose:**  
  - Train models, save them, and evaluate on test/sample data, printing evaluation metrics.
- **How to use:**  
  - Enable/disable each function by commenting/uncommenting in `main`.

---

## Advanced Steps

### Cross-Validation and Manual 10-Fold Evaluation
- **Functions:**  
  - `runModelFold(path)`
  - `runManual10FoldDataClean(dataPath, seed)`
  - `runManual10FoldDataGroup(dataPath, seed)`
  - `runManual10FoldDataCombineRF(dataPath, seed)`
  - `runManual10FoldDataCombineMP(dataPath, seed)`
- **Purpose:**  
  - Perform 10-fold cross-validation using Weka or custom pipeline for robust model comparison.
- **How to use:**  
  - Call the appropriate function for your data and model type. Results are printed to the console.

---

## Visualization with `draw.ipynb`

- **Purpose:**  
  `draw.ipynb` is a Jupyter Notebook for visualizing your data and model results. You can use it to:
  - Plot feature distributions, correlations, and outlier detection
  - Visualize model predictions vs. ground truth
  - Compare model performances with charts (bar, boxplot, etc.)

- **How to use:**
  1. Make sure you have Python 3 and Jupyter installed (`pip install jupyter pandas matplotlib seaborn scipy scikit-learn liac-arff`).
  2. Place `draw.ipynb` in the project folder.
  3. Run `jupyter notebook draw.ipynb` from your terminal.
  4. Follow the notebook instructions to load ARFF files or result files and generate visualizations.

---

## Summary Table

| Step                | Java Function(s)                  | Purpose                                      | Output/Usage                |
|---------------------|-----------------------------------|----------------------------------------------|-----------------------------|
| 1. Data Conversion     | `convertAllCsvToArff`             | CSV → ARFF                                   | ARFF files                  |
| 2. Data Integration    | `processFiles`, `processFileCombine` | Merge, feature engineering                | Combined ARFF files         |
| 3. Preprocessing & Split | `preprocess*`                  | Clean, normalize, cluster/group, outlier     | Cleaned ARFF files          |
| 4. Feature Aggregation | `aggregateMinuteFeaturesAndSave`  | Aggregate minute-level to feature-level      | Aggregated ARFF files       |
| 5. Model Training & Evaluation | `buildTestModel`, `buildModel`, `testModel`, etc. | Train, evaluate, and test models | Metrics, results, models    |
| Cross-Validation    | `runModelFold`, `runManual10Fold*` | 10-fold cross-validation                     | Metrics, results, models in `model_10Fold/` |
| Visualization       | `draw.ipynb`                      | Data/result visualization (Python notebook)   | Plots, charts               |

---

## Dataset Download

- **Dataset used in this project:**  
  https://www.kaggle.com/datasets/podsyp/production-quality

---

## About

- **Project:** Data Mining for Coffee Roasting Quality Prediction
- **Authors:** Huỳnh Hữu Nghĩa

---

**Tip:**  
You can comment/uncomment steps in `Data_MiningProject.java` to run only the parts you want, making it easy to debug or experiment with each stage of the pipeline.
