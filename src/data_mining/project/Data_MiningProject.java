package data_mining.project;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import weka.core.DenseInstance;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.SimpleKMeans;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.classifiers.Evaluation;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.Attribute;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Data_MiningProject {

    static final int WINDOW_SIZE = 60;

    public static void main(String[] args) throws Exception {

        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter the path to data_X.csv:");
        String pathX = scanner.nextLine();

        if (!isFileValid(pathX)) {
            System.out.println("❌ File data_X.csv does not exist or the path is invalid.");
            return;
        }

        System.out.println("Enter the path to data_Y.csv:");
        String pathY = scanner.nextLine();

        if (!isFileValid(pathY)) {
            System.out.println("❌ File data_Y.csv does not exist or the path is invalid.");
            return;
        }

        System.out.println("Enter the path to sample_submission.csv:");
        String pathSubmission = scanner.nextLine();

        if (!isFileValid(pathSubmission)) {
            System.out.println("❌ File sample_submission.csv does not exist or the path is invalid.");
            return;
        }

        // Step 1: Convert CSV files to ARFF format
        // convertAllCsvToArff("src/resources/original_data/data_X.csv",
        //         "src/resources/original_data/data_Y.csv",
        //         "src/resources/original_data/sample_submission.csv");
        convertAllCsvToArff(pathX,pathY,pathSubmission);

        // Perform data integration processing
        // processFiles("src/resources/original_data/data_X.csv",
        //         "src/resources/original_data/data_Y.csv",
        //         "src/resources/original_data/sample_submission.csv");
        processFiles(pathX,pathY,pathSubmission);


        // Perform combined data processing
        // processFileCombine("src/resources/original_data/data_X.csv",
        //         "src/resources/original_data/data_Y.csv",
        //         "src/resources/original_data/sample_submission.csv");
        processFileCombine(pathX,pathY,pathSubmission);
   

        // Step 2: Split train, test, and clean data train, test, and sample
        // Read ARFF file paths (you can hard-code them or use command-line arguments)
        String filePathData = "src/resources/processed_data/data.arff";
        String filePathSample = "src/resources/processed_data/data_sample.arff";

        // 1. Load data
        Instances data = loadData(filePathData);
        Instances sample = loadData(filePathSample);

        // 2. Shuffle the data (optional)
        data.randomize(new java.util.Random(42));

        // 3. Split into 70/30 for training and testing
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;

        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);

        // Clean the data
        PreprocessingModel modelerClean = new PreprocessingModel();
        Instances trainDataClean = preprocessTrainDataClean(trainData, modelerClean);
        Instances testDataClean = preprocessTestDataClean(testData, modelerClean);
        Instances sampleData = preprocessTestDataClean(sample, modelerClean);

        // Save cleaned datasets
        saveData(trainDataClean, "src/resources/processed_data/data_train_clean.arff");
        saveData(testDataClean, "src/resources/processed_data/data_test_clean.arff");
        saveData(sampleData, "src/resources/processed_data/sample_clean.arff");

        // Clean + Cluster
        PreprocessingModel modelerCluster = new PreprocessingModel();
        Instances trainDataCluster = preprocessTrainDataCleanCluster(trainData, modelerCluster);
        Instances testDataCluster = preprocessTestDataCleanCluster(testData, modelerCluster);
        Instances sampleDataCluster = preprocessTestDataCleanCluster(sample, modelerCluster);

        // Save
        saveData(trainDataCluster, "src/resources/processed_data/data_train_clean_Cluster.arff");
        saveData(testDataCluster, "src/resources/processed_data/data_test_clean_Cluster.arff");
        saveData(sampleDataCluster, "src/resources/processed_data/sample_clean_Cluster.arff");


        // Combine data
        String filePathCombine = "src/resources/processed_data/dataCombine.arff";
        String filePathCombineSample = "src/resources/processed_data/dataCombine_sample.arff";

        // Load combined data
        Instances dataCombine = loadData(filePathCombine);
        Instances sampleCombine = loadData(filePathCombineSample);

        // Shuffle
        dataCombine.randomize(new java.util.Random(42));

        // Split combined data
        int trainSizeCombine = (int) Math.round(dataCombine.numInstances() * 0.7);
        int testSizeCombine = dataCombine.numInstances() - trainSize;

        Instances trainDataCombine = new Instances(dataCombine, 0, trainSizeCombine);
        Instances testDataCombine = new Instances(dataCombine, trainSize, testSizeCombine);

        // Group preprocessing with outlier removed
        PreprocessingModel modelerGroupOut = new PreprocessingModel();
        Instances trainDataGroupOut = preprocessTrainDataGroupRemoveOutliers(trainDataCombine, modelerGroupOut);
        Instances testDataGroupOut = preprocessTestDataGroupRemoveOutliers(testDataCombine, modelerGroupOut);
        Instances sampleDataGroupOut = preprocessTestDataGroupRemoveOutliers(sampleCombine, modelerGroupOut);

        // Save cleaned group with outliers removed
        saveData(trainDataGroupOut , "src/resources/processed_data/data_train_Group_Out_clean.arff");
        saveData(testDataGroupOut, "src/resources/processed_data/data_test_Group_Out_clean.arff");
        saveData(sampleDataGroupOut, "src/resources/processed_data/sample_Group_Out_clean.arff");

        // Combine data form data clean
        aggregateMinuteFeaturesAndSave(
            "src/resources/processed_data/data_train_clean.arff",
            "src/resources/processed_data/dataCombine_train_clean.arff"
        );
        
        aggregateMinuteFeaturesAndSave(
            "src/resources/processed_data/data_test_clean.arff",
            "src/resources/processed_data/dataCombine_test_clean.arff"
        );
        
        aggregateMinuteFeaturesAndSave(
            "src/resources/processed_data/sample_clean.arff",
            "src/resources/processed_data/sampleCombine_clean.arff"
        );


        // Step 3: build on train data and evaluate models on sample data
        // Train evaluation
        // buildModel("src/resources/processed_data/data_train_clean.arff",
        //         "data_clean");
        
        // buildModelCluster("src/resources/processed_data/data_train_clean_Cluster.arff",
        //         "data_clean_Cluster");

        // buildModel("src/resources/processed_data/data_train_Group_Out_clean.arff",
        //         "data_Group_Out_clean");


        // buildModel("src/resources/processed_data/dataCombine_train_clean.arff",
        //         "dataCombine");



        // Train + Test evaluation
        buildTestModel("src/resources/processed_data/data_train_clean.arff",
                "src/resources/processed_data/data_test_clean.arff",
                "data_clean");
        
        buildTestModelCluster("src/resources/processed_data/data_train_clean_Cluster.arff",
                "src/resources/processed_data/data_test_clean_Cluster.arff",
                "data_clean_Cluster");

        buildTestModel("src/resources/processed_data/data_train_Group_Out_clean.arff",
                "src/resources/processed_data/data_test_Group_Out_clean.arff",
                "data_Group_Out_clean");
        
        buildTestModel("src/resources/processed_data/dataCombine_train_clean.arff",
                "src/resources/processed_data/dataCombine_test_clean.arff",
                "dataCombine");


        // Test evaluation
        // data clean
        // testModel("src/resources/model/IBk_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/RandomForest_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/LinearRegression_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/DecisionTable_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/M5Rules_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/M5P_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");
        // testModel("src/resources/model/MultilayerPerceptron_data_clean_model.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean");


        // data clean cluster
        // testModelCluster("src/resources/model/RandomForest_data_clean_Cluster_model.model",
        //         "src/resources/processed_data/data_test_clean_Cluster.arff",
        //         "data clean cluster");

        // data group clean out
        // testModel("src/resources/model/RandomForest_data_Group_Out_clean_model.model",
        //         "src/resources/processed_data/data_test_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("src/resources/model/MultilayerPerceptron_data_Group_Out_clean_model.model",
        //         "src/resources/processed_data/data_test_Group_Out_clean.arff",
        //         "data group clean remove outline");


        // dataCombine clean
        // testModel("src/resources/model/RandomForest_dataCombine_model.model",
        //         "src/resources/processed_data/dataCombine_test_clean.arff",
        //         "dataCombine clean");
        // testModel("src/resources/model/MultilayerPerceptron_dataCombine_model.model",
        //         "src/resources/processed_data/dataCombine_test_clean.arff",
        //         "dataCombine clean");

        // Step 4: Cross-validation for comparison
        // using weka to predict
        // runModelFold("src/resources/processed_data/data_train_clean.arff");
        // runModelFold("src/resources/processed_data/data_train_Group_Out_clean.arff");
        // runModelFold("src/resources/processed_data/dataCombine_train_clean.arff");

        // testModel("src/resources/model_10Fold/RandomForest_data_clean_model_10Fold.model",
        //         "src/resources/processed_data/data_test_clean.arff",
        //         "data clean 10-fold");

        // testModel("src/resources/model_10Fold/MultilayerPerceptron_data_Group_Out_clean_model_10Fold.model",
        //         "src/resources/processed_data/data_test_Group_Out_clean.arff",
        //         "data group clean remove outline 10-fold");

        // using our model to predict
        // runManual10FoldDataGroup("src/resources/processed_data/dataCombine.arff", 1);
        runManual10FoldDataClean("src/resources/processed_data/data.arff", 1);
        runManual10FoldDataCombineRF("src/resources/processed_data/data.arff",1);
        runManual10FoldDataCombineMP("src/resources/processed_data/data.arff",1);

        //Step 5: Test on sample
        // testModel("src/resources/model/RandomForest_data_Group_Out_clean_model.model",
        //             "src/resources/processed_data/sample_Group_Out-clean_model.arff",
        //             "sample data test test RandomForest");
        testModel("src/resources/model/RandomForest_data_clean_model.model",
                    "src/resources/processed_data/sample_clean.arff",
                    "sample data clean test RandomForest");
        testModel("src/resources/model/RandomForest_dataCombine_model.model",
                    "src/resources/processed_data/sampleCombine_clean.arff",
                    "sample dataCombine test RandomForest");
        testModel("src/resources/model/MultilayerPerceptron_dataCombine_model.model",
                    "src/resources/processed_data/sampleCombine_clean.arff",
                    "sample dataCombine test MultilayerPerceptron");
    }

    public static void aggregateMinuteFeaturesAndSave(String inputPath, String outputPath) throws Exception {
        Instances data = loadData(inputPath);
        Instances dataAggregated = aggregateMinuteFeatures(data);
        saveData(dataAggregated, outputPath);
    }

    public static boolean isFileValid(String path) {
        File file = new File(path);
        return file.exists() && file.isFile();
    }

    public static void runModelFold(String path) throws Exception{
        Instances data = loadData(path);

        Classifier[] models = {
            ModelFactory.getIBkModel(),
            ModelFactory.getRandomForestModel(),
            ModelFactory.getLinearRegressionModel(),
            ModelFactory.getDecisionTableModel(),
            ModelFactory.getM5RulesModel(),
            ModelFactory.getM5PModel(),
            ModelFactory.getMultilayerPerceptronModel()
        };
        
        for (Classifier model : models) {
            long startTime = System.currentTimeMillis();

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));

            long endTime = System.currentTimeMillis();

            long duration = endTime - startTime;

            System.out.println("=== 10-Fold Cross-Validation Result ===");

            System.out.println(eval.toSummaryString());
            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Error Rate: " + eval.errorRate());

            System.out.println("Time taken for 10-Fold Cross-Validation: " + duration + " milliseconds");
        }
    }

    public static void buildTestModel(String trainPath,String testPath,String file) throws Exception {
        Classifier[] models = {
            ModelFactory.getIBkModel(),
            ModelFactory.getRandomForestModel(),
            ModelFactory.getLinearRegressionModel(),
            ModelFactory.getDecisionTableModel(),
            ModelFactory.getM5RulesModel(),
            ModelFactory.getM5PModel(),
            ModelFactory.getMultilayerPerceptronModel()
        };


        Instances dataTrain = loadData(trainPath);
        Instances dataTest = loadData(testPath);

        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in trainData: " + dataTrain.size());
        System.out.println("Number of instances in testData: " + dataTest.size());

        for (Classifier model : models) {
            System.out.println("=== Evaluating model: " + model.getClass().getSimpleName() + " ===");
        
            long startTrainTime = System.currentTimeMillis();
        
            model.buildClassifier(dataTrain);
        
            long endTrainTime = System.currentTimeMillis();
            long trainingDuration = endTrainTime - startTrainTime;
        
            System.out.println("Time taken to train model: " + trainingDuration + " milliseconds");

            System.out.println("=== Evaluating model on test data ===");
            Evaluation eval = new Evaluation(dataTest);
            eval.evaluateModel(model, dataTest);
        
            System.out.println(eval.toSummaryString());
            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Error Rate: " + eval.errorRate()+"\n");

            String modelFilePath = "src/resources/model/" + model.getClass().getSimpleName() +"_"+ file + "_model.model";
            saveModel(model, modelFilePath);
        }

    }

    public static void buildTestModelCluster(String trainPath,String testPath,String file) throws Exception {
        Classifier[] models = {
            ModelFactory.getIBkModel(),
            ModelFactory.getRandomForestModel(),
            ModelFactory.getLinearRegressionModel(),
            ModelFactory.getDecisionTableModel(),
            ModelFactory.getM5RulesModel(),
            ModelFactory.getM5PModel(),
            ModelFactory.getMultilayerPerceptronModel()
        };


        Instances dataTrain = loadDataCluster(trainPath);
        Instances dataTest = loadDataCluster(testPath);

        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in trainData: " + dataTrain.size());
        System.out.println("Number of instances in testData: " + dataTest.size());

        for (Classifier model : models) {
            System.out.println("=== Evaluating model: " + model.getClass().getSimpleName() + " ===");
        
            long startTrainTime = System.currentTimeMillis();
        
            model.buildClassifier(dataTrain);
        
            long endTrainTime = System.currentTimeMillis();
            long trainingDuration = endTrainTime - startTrainTime;
        
            System.out.println("Time taken to train model: " + trainingDuration + " milliseconds");

            System.out.println("=== Evaluating model on test data ===");
            Evaluation eval = new Evaluation(dataTest);
            eval.evaluateModel(model, dataTest);
        
            System.out.println(eval.toSummaryString());
            System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Error Rate: " + eval.errorRate()+"\n");

            String modelFilePath = "src/resources/model/" + model.getClass().getSimpleName() +"_"+ file + "_model.model";
            saveModel(model, modelFilePath);
        }

    }

    public static void testModel(String pathMode,String testPath,String file) throws Exception {

        Classifier model = loadModel(pathMode);
        Instances dataTest = loadData(testPath);

        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in testData: " + dataTest.size());

        System.out.println("\n=== Evaluating model: " + model.getClass().getSimpleName() + " ===");

        System.out.println("=== Evaluating model on test data ===");
        Evaluation eval = new Evaluation(dataTest);
        eval.evaluateModel(model, dataTest);
        
        System.out.println(eval.toSummaryString());
        System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
        System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
        System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
        System.out.println("Error Rate: " + eval.errorRate());

    }

    public static void testModelCluster(String pathMode,String testPath,String file) throws Exception {

        Classifier model = loadModel(pathMode);
        Instances dataTest = loadDataCluster(testPath);

        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in testData: " + dataTest.size());

        System.out.println("\n=== Evaluating model: " + model.getClass().getSimpleName() + " ===");

        System.out.println("=== Evaluating model on test data ===");
        Evaluation eval = new Evaluation(dataTest);
        eval.evaluateModel(model, dataTest);
        
        System.out.println(eval.toSummaryString());
        System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
        System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
        System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
        System.out.println("Error Rate: " + eval.errorRate());

    }

    public static void buildModel(String trainPath,String file) throws Exception {
        Classifier[] models = {
            ModelFactory.getIBkModel(),
            ModelFactory.getRandomForestModel(),
            ModelFactory.getLinearRegressionModel(),
            ModelFactory.getDecisionTableModel(),
            ModelFactory.getM5RulesModel(),
            ModelFactory.getM5PModel(),
            ModelFactory.getMultilayerPerceptronModel()
        };

        Instances dataTrain = loadData(trainPath);
        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in trainData: " + dataTrain.size());

        for (Classifier model : models) {
            System.out.println("=== Evaluating model: " + model.getClass().getSimpleName() + " ===");
        
            long startTrainTime = System.currentTimeMillis();
        
            model.buildClassifier(dataTrain);
        
            long endTrainTime = System.currentTimeMillis();
            long trainingDuration = endTrainTime - startTrainTime;
        
            System.out.println("Time taken to train model: " + trainingDuration + " milliseconds\n");

            String modelFilePath = "src/resources/model/" + model.getClass().getSimpleName() +"_"+ file +"_model.model";
            saveModel(model, modelFilePath);
        }

    }

    public static void buildModelCluster(String trainPath,String file) throws Exception {
        Classifier[] models = {
            ModelFactory.getIBkModel(),
            ModelFactory.getRandomForestModel(),
            ModelFactory.getLinearRegressionModel(),
            ModelFactory.getDecisionTableModel(),
            ModelFactory.getM5RulesModel(),
            ModelFactory.getM5PModel(),
            ModelFactory.getMultilayerPerceptronModel()
        };

        Instances dataTrain = loadDataCluster(trainPath);
        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in trainData: " + dataTrain.size());

        for (Classifier model : models) {
            System.out.println("=== Evaluating model: " + model.getClass().getSimpleName() + " ===");
        
            long startTrainTime = System.currentTimeMillis();
        
            model.buildClassifier(dataTrain);
        
            long endTrainTime = System.currentTimeMillis();
            long trainingDuration = endTrainTime - startTrainTime;
        
            System.out.println("Time taken to train model: " + trainingDuration + " milliseconds\n");

            String modelFilePath = "src/resources/model/" + model.getClass().getSimpleName() +"_"+ file +"_model.model";
            saveModel(model, modelFilePath);
        }

    }
    

    public static void convertAllCsvToArff(String inputPathX,String inputPathY,String inputSample) throws Exception {

        final String outputArffX = "src/resources/processed_data/data_X.arff";
        saveDataXToArff(inputPathX, outputArffX);

        final String outputArffY = "src/resources/processed_data/data_Y.arff";
        saveDataYToArff(inputPathY, outputArffY);

        final String outputArffSample = "src/resources/processed_data/sample_submission.arff";
        saveDataYToArff(inputSample, outputArffSample);
    }


    public static Instances alignAttributes(Instances source, Instances target) throws Exception {
        ArrayList<Integer> toKeep = new ArrayList<>();
        
        for (int i = 0; i < source.numAttributes(); i++) {
            String name = source.attribute(i).name();
            Attribute targetAttr = target.attribute(name);
            if (targetAttr != null) {
                toKeep.add(targetAttr.index());
            }
        }

        StringBuilder indicesStr = new StringBuilder();
        for (int i = 0; i < target.numAttributes(); i++) {
            if (!toKeep.contains(i)) {
                indicesStr.append((i + 1)).append(",");
            }
        }

        if (!indicesStr.toString().isEmpty()) {
            Remove remove = new Remove();
            remove.setAttributeIndices(indicesStr.toString());
            remove.setInputFormat(target);
            target = Filter.useFilter(target, remove);
        }

        return target;
    }

    public static Classifier loadModel(String modelFilePath) throws Exception {
        FileInputStream fileIn = new FileInputStream(modelFilePath);
        ObjectInputStream objectIn = new ObjectInputStream(fileIn);
        Classifier model = (Classifier) objectIn.readObject();
        objectIn.close();
        return model;
    }

    public static class OutlierThresholds {
        public Map<Integer, double[]> bounds = new HashMap<>();
    }

    public static class PreprocessingModel {
        public Normalize normalizeFilter;
        public OutlierThresholds outlierThresholds;

        public SimpleKMeans kMeansModel;
        public AddCluster addClusterFilter;

        public PreprocessingModel() {
            normalizeFilter = new Normalize();
        }
    }

    public static Instances preprocessTrainDataClean(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = replaceOutliersWithMeanTrain(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        return data;
    }

    public static Instances preprocessTestDataClean(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = replaceOutliersWithMeanTest(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        return data;
    }

    public static Instances preprocessTrainDataCleanCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = replaceOutliersWithMeanTrain(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(5);
        Instances dataWithoutClass = new Instances(data);
        dataWithoutClass.setClassIndex(-1);
        kMeans.buildClusterer(dataWithoutClass);

        AddCluster addCluster = new AddCluster();
        addCluster.setClusterer(kMeans);
        addCluster.setInputFormat(data);
        data = Filter.useFilter(data, addCluster);

        model.kMeansModel = kMeans;
        model.addClusterFilter = addCluster;

        return data;
    }

    public static Instances preprocessTestDataCleanCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = replaceOutliersWithMeanTest(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        model.addClusterFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.addClusterFilter);

        return data;
    }

    public static Instances preprocessTrainDataGroup(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        return data;
    }

    public static Instances preprocessTestDataGroup(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        return data;
    }

    public static Instances preprocessTrainDataGroupRemoveOutliers(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = replaceOutliersWithMeanTrain(data, model.outlierThresholds);

        return data;
    }

    public static Instances preprocessTestDataGroupRemoveOutliers(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = replaceOutliersWithMeanTest(data, model.outlierThresholds);

        return data;
    }

    public static Instances preprocessTrainDataGroupRemoveOutliersCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = replaceOutliersWithMeanTrain(data, model.outlierThresholds);

        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(5);
        Instances dataWithoutClass = new Instances(data);
        dataWithoutClass.setClassIndex(-1);
        kMeans.buildClusterer(dataWithoutClass);

        AddCluster addCluster = new AddCluster();
        addCluster.setClusterer(kMeans);
        addCluster.setInputFormat(data);
        data = Filter.useFilter(data, addCluster);

        model.kMeansModel = kMeans;
        model.addClusterFilter = addCluster;

        return data;
    }

    public static Instances preprocessTestDataGroupRemoveOutliersCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = replaceOutliersWithMeanTest(data, model.outlierThresholds);

        model.addClusterFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.addClusterFilter);

        return data;
    }

    public static OutlierThresholds calculateOutlierThresholds(Instances data) {
        OutlierThresholds thresholds = new OutlierThresholds();
        int classIndex = data.classIndex();

        for (int i = 0; i < data.numAttributes(); i++) {
            if (i == classIndex) continue;
            Attribute attr = data.attribute(i);
            if (attr.isNumeric()) {
                List<Double> values = new ArrayList<>();
                for (int j = 0; j < data.numInstances(); j++) {
                    values.add(data.instance(j).value(i));
                }

                Collections.sort(values);
                int n = values.size();
                double Q1 = values.get(n / 4);
                double Q3 = values.get(3 * n / 4);
                double IQR = Q3 - Q1;
                double lowerBound = Q1 - 1.5 * IQR;
                double upperBound = Q3 + 1.5 * IQR;

                thresholds.bounds.put(i, new double[]{lowerBound, upperBound});
            }
        }

        return thresholds;
    }

    // public static Instances removeOutliersTrain(Instances data, OutlierThresholds thresholds) {
    //     int classIndex = data.classIndex();

    //     for (int i = data.numInstances() - 1; i >= 0; i--) {
    //         Instance instance = data.instance(i);
    //         for (int attrIndex : thresholds.bounds.keySet()) {
    //             if (attrIndex == classIndex) continue;
    //             double val = instance.value(attrIndex);
    //             double[] bound = thresholds.bounds.get(attrIndex);
    //             if (val < bound[0] || val > bound[1]) {
    //                 data.delete(i);
    //                 break;
    //             }
    //         }
    //     }

    //     data.setClassIndex(classIndex);
    //     return data;
    // }

    public static Map<Integer, Double> calculateAttributeMeans(Instances data) {
        Map<Integer, Double> means = new HashMap<>();
        int classIndex = data.classIndex();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i == classIndex) continue;
            Attribute attr = data.attribute(i);
            if (attr.isNumeric()) {
                double sum = 0;
                int count = 0;
                for (int j = 0; j < data.numInstances(); j++) {
                    double val = data.instance(j).value(i);
                    if (!Double.isNaN(val)) {
                        sum += val;
                        count++;
                    }
                }
                if (count > 0) {
                    means.put(i, sum / count);
                }
            }
        }
        return means;
    }

    public static Instances replaceOutliersWithMeanTrain(Instances data, OutlierThresholds thresholds) {
        int classIndex = data.classIndex();
        Map<Integer, Double> means = calculateAttributeMeans(data);

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            for (int attrIndex : thresholds.bounds.keySet()) {
                if (attrIndex == classIndex) continue;
                double val = instance.value(attrIndex);
                double[] bound = thresholds.bounds.get(attrIndex);
                if (val < bound[0] || val > bound[1]) {
                    Double mean = means.get(attrIndex);
                    if (mean != null) {
                        instance.setValue(attrIndex, mean);
                    }
                }
            }
        }

        data.setClassIndex(classIndex);
        return data;
    }

    public static Instances replaceOutliersWithMeanTest(Instances data, OutlierThresholds thresholds) {
        int classIndex = data.classIndex();
        Map<Integer, Double> means = calculateAttributeMeans(data);

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            for (int attrIndex : thresholds.bounds.keySet()) {
                if (attrIndex == classIndex) continue;
                double val = instance.value(attrIndex);
                double[] bound = thresholds.bounds.get(attrIndex);
                if (val < bound[0] || val > bound[1]) {
                    Double mean = means.get(attrIndex);
                    if (mean != null) {
                        instance.setValue(attrIndex, mean);
                    }
                }
            }
        }

        data.setClassIndex(classIndex);
        return data;
    }

    public static Instances loadData(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1); 
        }
        return data;
    }

    public static Instances loadDataCluster(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 2); 
        }
        return data;
    }

    public static void saveModel(Classifier model, String modelFilePath) throws Exception {
        SerializationHelper.write(modelFilePath, model);
        System.out.println("Model saved to: " + modelFilePath);
    }

    public static void processFiles(String dataXPath,String dataYPath,String samplePath ) throws Exception {

        String[][] dataX = readData(dataXPath);
        String[][] dataY = readData(dataYPath);
        String[][] sample = readData(samplePath);

        String outputArffPath = "src/resources/processed_data/data.arff";

        BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputArffPath));
        writeArffHeader(writer);

        writeData(writer,dataX, dataY);

        writer.close();

        String outputArffPathSample = "src/resources/processed_data/data_sample.arff";

        BufferedWriter writerSample = Files.newBufferedWriter(Paths.get(outputArffPathSample));
        writeArffHeader(writerSample);

        writeData(writerSample,dataX, sample);

        writerSample.close();
    }

    public static String[][] readData(String path) {
        List<String[]> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",", -1);
                dataList.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataList.toArray(new String[0][]);
    }

    private static void writeArffHeader(BufferedWriter writer) throws IOException {
        writer.write("@RELATION roasting_quality\n\n");

        // writer.write("@ATTRIBUTE data_time DATE \"yyyy-MM-dd'T'HH:mm:ss\"\n");
    
        for (int k = 1; k <= WINDOW_SIZE; k++) {
            for (int i = 1; i <= 5; i++) {
                for (int j = 1; j <= 3; j++) {
                    writer.write(String.format("@ATTRIBUTE T_data_%d_%d_minute_%02d NUMERIC\n", i, j, k));
                    
                }
            }
            writer.write(String.format("@ATTRIBUTE H_data_minute_%02d NUMERIC\n", k));
            
        }
    
        writer.write(String.format("@ATTRIBUTE AH_data NUMERIC\n"));
        writer.write("@ATTRIBUTE quality NUMERIC\n\n");
        writer.write("@DATA\n");
    }

    private static void writeData(BufferedWriter writer, String[][] dataX, String[][] dataY) throws IOException {
        Map<String, String[]> dataMapX = new HashMap<>();
        for (String[] row : dataX) {
            dataMapX.put(row[0], row);
        }
    
        dataY = Arrays.copyOfRange(dataY, 1, dataY.length);

        for (String[] rowY : dataY) {

            String timestamp = rowY[0];
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

            LocalDateTime dateTime = LocalDateTime.parse(timestamp, formatter);
            LocalDateTime rounded = dateTime.withMinute(0).withSecond(0).withNano(0);
            String roundedStr = rounded.format(formatter);

            // writer.write(roundedStr.replace(" ", "T")+ ",");

            String quality = rowY[1];

            String AH = null;
            for (int k = WINDOW_SIZE; k >= 1; k--) {
                String[] data = dataMapX.get(DataCount(roundedStr, k));
                if (data != null) {
                    for (int j = 1; j < data.length-1; j++) {
                        writer.write(data[j] + ",");
                        
                    }
                } else {
                    for (int j = 1; j <= 16; j++) {
                        writer.write("?,");
                
                    }
                }
                if (k == WINDOW_SIZE) {
                    AH = data[data.length-1];
                }
                else {
                    if(AH.compareTo(data[data.length - 1]) != 0){
                        System.out.println("AH different AH " + AH + " - " + data[data.length-1]);
                    }
                }
            }
            writer.write(AH + ",");

            writer.write(quality + "\n");
        }
    }
    
    private static void writeArffHeaderY(BufferedWriter writer) throws IOException {
        writer.write("@RELATION dataY\n\n");
        writer.write("@ATTRIBUTE timestone DATE \"yyyy-MM-dd'T'HH:mm:ss\"\n");
        writer.write("@ATTRIBUTE quality NUMERIC\n\n");
        writer.write("@DATA\n");
    }
    
    private static void writeDataY(BufferedWriter writer, String[][] dataY) throws IOException {
        dataY = Arrays.copyOfRange(dataY, 1, dataY.length);
        DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");

        for (String[] row : dataY) {
            LocalDateTime dateTime = LocalDateTime.parse(row[0], inputFormatter);
            String formattedTimestamp = dateTime.format(outputFormatter);
            writer.write(formattedTimestamp + "," + row[1] + "\n");
        }
    }
    
    public static void saveDataYToArff(String inputPathY, String outputArffY) {
        try {
            String[][] dataY = readData(inputPathY);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputArffY))) {
                writeArffHeaderY(writer);
                writeDataY(writer, dataY);
            }
            System.out.println("Saved dataY to: " + outputArffY);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeArffHeaderX(BufferedWriter writer) throws IOException {
        writer.write("@RELATION dataX\n\n");
        writer.write("@ATTRIBUTE timestone DATE \"yyyy-MM-dd'T'HH:mm:ss\"\n");

        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= 3; j++) {
                writer.write(String.format("@ATTRIBUTE T_data_%d_%d NUMERIC\n", i, j));
            }
        }
        writer.write("@ATTRIBUTE H_data NUMERIC\n");
        writer.write("@ATTRIBUTE AH_data NUMERIC\n\n");
        writer.write("@DATA\n");
    }
    
    private static void writeDataX(BufferedWriter writer, String[][] dataX) throws IOException {
        dataX = Arrays.copyOfRange(dataX, 1, dataX.length);
        DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");

        for (String[] row : dataX) {
            LocalDateTime dateTime = LocalDateTime.parse(row[0], inputFormatter);
            String formattedTimestamp = dateTime.format(outputFormatter);
            writer.write(formattedTimestamp + ",");

            for (int i = 1; i < row.length; i++) {
                if(i == (row.length - 1)) {
                    writer.write(row[i]);
                }
                else {
                    writer.write(row[i] + ",");
                }
            }
            writer.write("\n");
        }
    }
    
    public static void saveDataXToArff(String inputPathX, String outputArffX) {
        try {
            String[][] dataX = readData(inputPathX);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputArffX))) {
                writeArffHeaderX(writer);
                writeDataX(writer, dataX);
            }
            System.out.println("Saved dataX to: " + outputArffX);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String DataCount(String data, int time) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        LocalDateTime dateTime = LocalDateTime.parse(data.trim(), formatter);
        LocalDateTime result = dateTime.minusMinutes(time);
        return result.format(formatter);
    }
    
    public static void saveData(Instances data, String outputFilePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new java.io.File(outputFilePath)); 
        saver.writeBatch(); 
    }


    public static void processFileCombine(String dataXPath,String dataYPath,String samplePath) throws Exception {

        String[][] dataX = readData(dataXPath);
        String[][] dataY = readData(dataYPath);
        String[][] sample = readData(samplePath);

        String outputArffPath = "src/resources/processed_data/dataCombine.arff";

        BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputArffPath));
        writeArffHeaderCombine(writer);

        writeDataCombine(writer,dataX, dataY);

        writer.close();

        String outputArffPathSample = "src/resources/processed_data/dataCombine_sample.arff";

        BufferedWriter writerSample = Files.newBufferedWriter(Paths.get(outputArffPathSample));
        writeArffHeaderCombine(writerSample);

        writeDataCombine(writerSample,dataX, sample);

        writerSample.close();
    }


    private static void writeArffHeaderCombine(BufferedWriter writer) throws IOException {
        writer.write("@RELATION roasting_quality_aggregated\n\n");

        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= 3; j++) {
                String base = String.format("T_data_%d_%d", i, j);
                writer.write(String.format("@ATTRIBUTE %s_start NUMERIC\n", base));
                writer.write(String.format("@ATTRIBUTE %s_end NUMERIC\n", base));
                writer.write(String.format("@ATTRIBUTE %s_mean NUMERIC\n", base));
                writer.write(String.format("@ATTRIBUTE %s_max NUMERIC\n", base));
                writer.write(String.format("@ATTRIBUTE %s_min NUMERIC\n", base));
                writer.write(String.format("@ATTRIBUTE %s_delta NUMERIC\n", base));
            }
        }

        String base = "H_data";
        writer.write(String.format("@ATTRIBUTE %s_start NUMERIC\n", base));
        writer.write(String.format("@ATTRIBUTE %s_end NUMERIC\n", base));
        writer.write(String.format("@ATTRIBUTE %s_mean NUMERIC\n", base));
        writer.write(String.format("@ATTRIBUTE %s_max NUMERIC\n", base));
        writer.write(String.format("@ATTRIBUTE %s_min NUMERIC\n", base));
        writer.write(String.format("@ATTRIBUTE %s_delta NUMERIC\n", base));

        writer.write("@ATTRIBUTE AH_data NUMERIC\n");
        writer.write("@ATTRIBUTE quality NUMERIC\n\n");
        writer.write("@DATA\n");
    }


    private static void writeDataCombine(BufferedWriter writer, String[][] dataX, String[][] dataY) throws IOException {
        Map<String, String[]> dataMapX = new HashMap<>();
        for (String[] row : dataX) {
            dataMapX.put(row[0], row);
        }
    
        dataY = Arrays.copyOfRange(dataY, 1, dataY.length);

        for (String[] rowY : dataY) {

            String timestamp = rowY[0];
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

            LocalDateTime dateTime = LocalDateTime.parse(timestamp, formatter);
            LocalDateTime rounded = dateTime.withMinute(0).withSecond(0).withNano(0);
            String roundedStr = rounded.format(formatter);


            String quality = rowY[1];

            Double[][] sensorData = new Double[60][17];

            for (int k = WINDOW_SIZE; k >= 1; k--) {
                String[] data = dataMapX.get(DataCount(roundedStr, k));

                if (data != null) {
                    for (int i = 1; i < data.length; i++) {
                        sensorData[WINDOW_SIZE-k][i - 1] = Double.parseDouble(data[i]);
                    }
                }
            }

            for (int i = 0; i < 16; i++) {
                Double start = null;
                Double end = null;
                double sum = 0;
                int count = 0;
                Double min = null;
                Double max = null;
                double deltaSum = 0;
                Double prev = null;

                for (int j = 0; j < WINDOW_SIZE; j++) {
                    Double value = sensorData[j][i];

                    if (value != null) {
                        if (start == null) start = value;
                        end = value;
                        sum += value;
                        count++;

                        if (min == null || value < min) min = value;
                        if (max == null || value > max) max = value;

                        if (prev != null) {
                            deltaSum += Math.abs(value - prev);
                        }
                        prev = value;
                    }
                }

                if (start == null) {
                    writer.write("?, ?, ?, ?, ?, ?,");

                } else {
                    double mean = sum / count;
                    double delta = deltaSum / Math.max(1, count - 1);

                    writer.write(String.format("%.2f,%.2f,%.6f,%.2f,%.2f,%.6f,", start, end, mean, max, min, delta));
                }
            }


            double sumAH = 0;
            int countAH = 0;

            for (int i = 0; i < WINDOW_SIZE; i++) {
                Double ahValue = sensorData[i][16];
                if (ahValue != null) {
                    sumAH += ahValue;
                    countAH++;
                }
            }

            if (countAH > 0) {
                double averageAH = sumAH / countAH;
                writer.write(String.format("%.2f,", averageAH));
            } else {
                writer.write("?,");
            }


            writer.write(quality + "\n");

        }
    }

    public static Instances aggregateMinuteFeatures(Instances data) throws Exception {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= 3; j++) {
                String prefix = String.format("T_data_%d_%d_minute_", i, j);
                attributes.add(new Attribute(String.format("T_data_%d_%d_start", i, j)));
                attributes.add(new Attribute(String.format("T_data_%d_%d_end", i, j)));
                attributes.add(new Attribute(String.format("T_data_%d_%d_mean", i, j)));
                attributes.add(new Attribute(String.format("T_data_%d_%d_max", i, j)));
                attributes.add(new Attribute(String.format("T_data_%d_%d_min", i, j)));
                attributes.add(new Attribute(String.format("T_data_%d_%d_delta", i, j)));
            }
        }
        attributes.add(new Attribute("H_data_start"));
        attributes.add(new Attribute("H_data_end"));
        attributes.add(new Attribute("H_data_mean"));
        attributes.add(new Attribute("H_data_max"));
        attributes.add(new Attribute("H_data_min"));
        attributes.add(new Attribute("H_data_delta"));

        attributes.add(new Attribute("AH_data"));

        Attribute qualityAttr = data.attribute("quality");
        attributes.add(new Attribute("quality"));

        Instances result = new Instances("roasting_quality_aggregated", attributes, data.numInstances());
        result.setClassIndex(result.numAttributes() - 1);

        for (int row = 0; row < data.numInstances(); row++) {
            double[] vals = new double[result.numAttributes()];
            int idx = 0;

            for (int i = 1; i <= 5; i++) {
                for (int j = 1; j <= 3; j++) {

                    double[] values = new double[60];
                    for (int k = 1; k <= 60; k++) {
                        String colName = String.format("T_data_%d_%d_minute_%02d", i, j, k);
                        Attribute attr = data.attribute(colName);
                        values[k - 1] = data.instance(row).value(attr);
                    }
                    vals[idx++] = values[0]; 
                    vals[idx++] = values[59]; 
                    vals[idx++] = Arrays.stream(values).average().orElse(Double.NaN);
                    vals[idx++] = Arrays.stream(values).max().orElse(Double.NaN);
                    vals[idx++] = Arrays.stream(values).min().orElse(Double.NaN);

                    double delta = 0;
                    for (int d = 1; d < values.length; d++) {
                        delta += Math.abs(values[d] - values[d - 1]);
                    }
                    vals[idx++] = delta / (values.length - 1);
                }
            }

            double[] hValues = new double[60];
            for (int k = 1; k <= 60; k++) {
                String colName = String.format("H_data_minute_%02d", k);
                Attribute attr = data.attribute(colName);
                hValues[k - 1] = data.instance(row).value(attr);
            }
            vals[idx++] = hValues[0];
            vals[idx++] = hValues[59];
            vals[idx++] = Arrays.stream(hValues).average().orElse(Double.NaN);
            vals[idx++] = Arrays.stream(hValues).max().orElse(Double.NaN);
            vals[idx++] = Arrays.stream(hValues).min().orElse(Double.NaN);
            double hDelta = 0;
            for (int d = 1; d < hValues.length; d++) {
                hDelta += Math.abs(hValues[d] - hValues[d - 1]);
            }
            vals[idx++] = hDelta / (hValues.length - 1);

            Attribute ahAttr = data.attribute("AH_data");
            vals[idx++] = data.instance(row).value(ahAttr);

            vals[idx++] = data.instance(row).value(qualityAttr);

            result.add(new DenseInstance(1.0, vals));
        }
        return result;
    }

    public static void runManual10FoldDataClean(String dataPath, int seed) throws Exception {
        Instances dataOriginal = loadData(dataPath);
        System.out.println("\n=== 10-fold CV on file: " + dataPath + " ===");
        System.out.println("Type: dataClean");
        System.out.println("Total instances: " + dataOriginal.numInstances());
        System.out.println("Model: " + "Random Forest");

        long startTime = System.currentTimeMillis();

        Instances data = new Instances(dataOriginal);
        data.randomize(new java.util.Random(seed));

        int numFolds = 10;
        double sumRmse = 0, sumMae = 0, sumCorr = 0, sumError = 0;

        for (int fold = 0; fold < numFolds; fold++) {
            Instances train = data.trainCV(numFolds, fold);
            Instances test = data.testCV(numFolds, fold);

            PreprocessingModel modelerClean = new PreprocessingModel();
            Instances trainDataClean = preprocessTrainDataClean(train, modelerClean);
            Instances testDataClean = preprocessTestDataClean(test, modelerClean);

            Classifier model = ModelFactory.getRandomForestModel();
            model.buildClassifier(trainDataClean);
            Evaluation eval = new Evaluation(trainDataClean);
            eval.evaluateModel(model, testDataClean);

            System.out.println("Fold " + (fold + 1) + ":");
            System.out.println("  RMSE: " + eval.rootMeanSquaredError());
            System.out.println("  MAE: " + eval.meanAbsoluteError());
            System.out.println("  Corr: " + eval.correlationCoefficient());
            System.out.println("  Error: " + eval.errorRate());

            sumRmse += eval.rootMeanSquaredError();
            sumMae += eval.meanAbsoluteError();
            sumCorr += eval.correlationCoefficient();
            sumError += eval.errorRate();
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        System.out.println("\n=== Average over 10 folds ===");
        System.out.println("Average Correlation coefficient: " + (sumCorr / numFolds));
        System.out.println("Average Root Mean Squared Error (RMSE): " + (sumRmse / numFolds));
        System.out.println("Average Mean Absolute Error (MAE): " + (sumMae / numFolds));
        System.out.println("Average Error Rate: " + (sumError / numFolds));
        System.out.println("Total time for 10-fold CV: " + duration + " milliseconds");
    }

    public static void runManual10FoldDataGroup(String dataPath, int seed) throws Exception {
        Instances dataOriginal = loadData(dataPath);
        System.out.println("\n=== 10-fold CV on file: " + dataPath + " ===");
        System.out.println("Type: dataGroup");
        System.out.println("Total instances: " + dataOriginal.numInstances());
        System.out.println("Model: " + "Random Forest");

        long startTime = System.currentTimeMillis();

        Instances data = new Instances(dataOriginal);
        data.randomize(new java.util.Random(seed));

        int numFolds = 10;
        double sumRmse = 0, sumMae = 0, sumCorr = 0, sumError = 0;

        for (int fold = 0; fold < numFolds; fold++) {
            Instances train = data.trainCV(numFolds, fold);
            Instances test = data.testCV(numFolds, fold);

            PreprocessingModel modelerClean = new PreprocessingModel();
            Instances trainDataClean = preprocessTrainDataGroupRemoveOutliers(train, modelerClean);
            Instances testDataClean = preprocessTrainDataGroupRemoveOutliers(test, modelerClean);

            Classifier model = ModelFactory.getRandomForestModel();
            model.buildClassifier(trainDataClean);
            Evaluation eval = new Evaluation(trainDataClean);
            eval.evaluateModel(model, testDataClean);

            System.out.println("Fold " + (fold + 1) + ":");
            System.out.println("  RMSE: " + eval.rootMeanSquaredError());
            System.out.println("  MAE: " + eval.meanAbsoluteError());
            System.out.println("  Corr: " + eval.correlationCoefficient());
            System.out.println("  Error: " + eval.errorRate());

            sumRmse += eval.rootMeanSquaredError();
            sumMae += eval.meanAbsoluteError();
            sumCorr += eval.correlationCoefficient();
            sumError += eval.errorRate();
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        System.out.println("\n=== Average over 10 folds ===");
        System.out.println("Average Correlation coefficient: " + (sumCorr / numFolds));
        System.out.println("Average Root Mean Squared Error (RMSE): " + (sumRmse / numFolds));
        System.out.println("Average Mean Absolute Error (MAE): " + (sumMae / numFolds));
        System.out.println("Average Error Rate: " + (sumError / numFolds));
        System.out.println("Total time for 10-fold CV: " + duration + " milliseconds");
    }

    public static void runManual10FoldDataCombineRF(String dataPath, int seed) throws Exception {
        Instances dataOriginal = loadData(dataPath);
        System.out.println("\n=== 10-fold CV on file: " + dataPath + " ===");
        System.out.println("Type: dataCombine");
        System.out.println("Total instances: " + dataOriginal.numInstances());
        System.out.println("Model: " + "Random Forest");

        long startTime = System.currentTimeMillis();

        Instances data = new Instances(dataOriginal);
        data.randomize(new java.util.Random(seed));

        int numFolds = 10;
        double sumRmse = 0, sumMae = 0, sumCorr = 0, sumError = 0;

        for (int fold = 0; fold < numFolds; fold++) {
            Instances train = data.trainCV(numFolds, fold);
            Instances test = data.testCV(numFolds, fold);

            PreprocessingModel modelerClean = new PreprocessingModel();
            Instances trainDataClean = aggregateMinuteFeatures(preprocessTrainDataClean(train, modelerClean));
            Instances testDataClean = aggregateMinuteFeatures(preprocessTestDataClean(test, modelerClean));

            Classifier model = ModelFactory.getRandomForestModel();
            model.buildClassifier(trainDataClean);
            Evaluation eval = new Evaluation(trainDataClean);
            eval.evaluateModel(model, testDataClean);

            System.out.println("Fold " + (fold + 1) + ":");
            System.out.println("  RMSE: " + eval.rootMeanSquaredError());
            System.out.println("  MAE: " + eval.meanAbsoluteError());
            System.out.println("  Corr: " + eval.correlationCoefficient());
            System.out.println("  Error: " + eval.errorRate());

            sumRmse += eval.rootMeanSquaredError();
            sumMae += eval.meanAbsoluteError();
            sumCorr += eval.correlationCoefficient();
            sumError += eval.errorRate();
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        System.out.println("\n=== Average over 10 folds ===");
        System.out.println("Average Correlation coefficient: " + (sumCorr / numFolds));
        System.out.println("Average Root Mean Squared Error (RMSE): " + (sumRmse / numFolds));
        System.out.println("Average Mean Absolute Error (MAE): " + (sumMae / numFolds));
        System.out.println("Average Error Rate: " + (sumError / numFolds));
        System.out.println("Total time for 10-fold CV: " + duration + " milliseconds");
    }

    public static void runManual10FoldDataCombineMP(String dataPath, int seed) throws Exception {
        Instances dataOriginal = loadData(dataPath);
        System.out.println("\n=== 10-fold CV on file: " + dataPath + " ===");
        System.out.println("Type: dataCombine");
        System.out.println("Total instances: " + dataOriginal.numInstances());
        System.out.println("Model: " + "MultilayerPerceptron");

        long startTime = System.currentTimeMillis();

        Instances data = new Instances(dataOriginal);
        data.randomize(new java.util.Random(seed));

        int numFolds = 10;
        double sumRmse = 0, sumMae = 0, sumCorr = 0, sumError = 0;

        for (int fold = 0; fold < numFolds; fold++) {
            Instances train = data.trainCV(numFolds, fold);
            Instances test = data.testCV(numFolds, fold);

            PreprocessingModel modelerClean = new PreprocessingModel();
            Instances trainDataClean = aggregateMinuteFeatures(preprocessTrainDataClean(train, modelerClean));
            Instances testDataClean = aggregateMinuteFeatures(preprocessTestDataClean(test, modelerClean));

            Classifier model = ModelFactory.getMultilayerPerceptronModel();
            model.buildClassifier(trainDataClean);
            Evaluation eval = new Evaluation(trainDataClean);
            eval.evaluateModel(model, testDataClean);

            System.out.println("Fold " + (fold + 1) + ":");
            System.out.println("  RMSE: " + eval.rootMeanSquaredError());
            System.out.println("  MAE: " + eval.meanAbsoluteError());
            System.out.println("  Corr: " + eval.correlationCoefficient());
            System.out.println("  Error: " + eval.errorRate());

            sumRmse += eval.rootMeanSquaredError();
            sumMae += eval.meanAbsoluteError();
            sumCorr += eval.correlationCoefficient();
            sumError += eval.errorRate();
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        System.out.println("\n=== Average over 10 folds ===");
        System.out.println("Average Correlation coefficient: " + (sumCorr / numFolds));
        System.out.println("Average Root Mean Squared Error (RMSE): " + (sumRmse / numFolds));
        System.out.println("Average Mean Absolute Error (MAE): " + (sumMae / numFolds));
        System.out.println("Average Error Rate: " + (sumError / numFolds));
        System.out.println("Total time for 10-fold CV: " + duration + " milliseconds");
    }
}
