package dmproject;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

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

public class Main {

    static final int WINDOW_SIZE = 60;

    public static void main(String[] args) throws Exception {
        // Step 1: 
        // Convert CSV files to ARFF format
        convertAllCsvToArff();

        // Perform data integration processing
        processFiles();

        // Perform combined data processing
        processFileCombine();




        // Step 2: Train, test, and clean data
        // Read ARFF file paths (you can hard-code them or use command-line arguments)
        String filePathData = "resources/processed_data/data.arff";
        String filePathSample = "resources/processed_data/data_sample.arff";

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
        saveData(trainDataClean, "resources/processed_data/data_train_clean.arff");
        saveData(testDataClean, "resources/processed_data/data_test_clean.arff");
        saveData(sampleData, "resources/processed_data/sample_clean.arff");

        // Clean + Cluster
        PreprocessingModel modelerCluster = new PreprocessingModel();
        Instances trainDataCluster = preprocessTrainDataClean(trainData, modelerCluster);
        Instances testDataCluster = preprocessTestDataClean(testData, modelerCluster);
        Instances sampleDataCluster = preprocessTestDataClean(sample, modelerCluster);

        // Save
        saveData(trainDataCluster, "resources/processed_data/data_train_clean_Cluster.arff");
        saveData(testDataCluster, "resources/processed_data/data_test_clean_Cluster.arff");
        saveData(sampleDataCluster, "resources/processed_data/sample_clean_Cluster.arff");


        // Combine data
        String filePathCombine = "resources/processed_data/dataCombine.arff";
        String filePathCombineSample = "resources/processed_data/dataCombine_sample.arff";

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

        // Group preprocessing
        PreprocessingModel modelerGroup = new PreprocessingModel();
        Instances trainDataGroup = preprocessTrainDataGroup(trainDataCombine, modelerGroup);
        Instances testDataGroup = preprocessTestDataGroup(testDataCombine, modelerGroup);
        Instances sampleDataGroup = preprocessTestDataGroup(sampleCombine, modelerGroup);

        // Save group-cleaned data
        saveData(trainDataGroup, "resources/processed_data/data_train_Group_clean.arff");
        saveData(testDataGroup, "resources/processed_data/data_test_Group_clean.arff");
        saveData(sampleDataGroup, "resources/processed_data/sample_Group_clean.arff");

        // Group preprocessing with outlier removal
        PreprocessingModel modelerGroupOut = new PreprocessingModel();
        Instances trainDataGroupOut = preprocessTrainDataGroupRemoveOutliers(trainDataCombine, modelerGroupOut);
        Instances testDataGroupOut = preprocessTestDataGroupRemoveOutliers(testDataCombine, modelerGroupOut);
        Instances sampleDataGroupOut = preprocessTestDataGroupRemoveOutliers(sampleCombine, modelerGroupOut);

        // Save cleaned group with outliers removed
        saveData(trainDataGroupOut , "resources/processed_data/data_train_Group_Out_clean.arff");
        saveData(testDataGroupOut, "resources/processed_data/data_test_Group_Out_clean.arff");
        saveData(sampleDataGroupOut, "resources/processed_data/sample_Group_Out_clean.arff");

        // clean data remove outliers cluster
        PreprocessingModel modelerGroupOutCluster = new PreprocessingModel();
        Instances trainDataGroupOutCluster = preprocessTrainDataGroupRemoveOutliers(trainDataCombine, modelerGroupOutCluster);
        Instances testDataGroupOutCluster = preprocessTestDataGroupRemoveOutliers(testDataCombine, modelerGroupOutCluster);
        Instances sampleDataGroupOutCluster = preprocessTestDataGroupRemoveOutliers(sampleCombine, modelerGroupOutCluster);

        saveData(trainDataGroupOutCluster , "resources/processed_data/data_train_Group_Out_Cluster_clean.arff");
        saveData(testDataGroupOutCluster, "resources/processed_data/data_test_Group_Out_Cluster_clean.arff");
        saveData(sampleDataGroupOutCluster , "resources/processed_data/sample_Group_Out_Cluster_clean.arff");




        // Step 3: Train and evaluate models
        buildModel("resources/processed_data/data_train_clean.arff",
                "data_clean");
        
        buildModel("resources/processed_data/data_train_clean_Cluster.arff",
                "data_clean_Cluster");

        buildModel("resources/processed_data/data_train_Group_clean.arff",
                "data_Group_clean");

        buildModel("resources/processed_data/data_train_Group_Out_clean.arff",
                "data_Group_Out_clean");

        buildModel("resources/processed_data/data_train_Group_Out_Cluster_clean.arff",
                "data_Group_Out_Cluster_clean");

        // Step 4:
        // Train + Test evaluation
        buildTestModel("resources/processed_data/data_train_clean.arff",
                "resources/processed_data/data_test_clean.arff",
                "resources/processed_data/sample_clean.arff",
                "data_clean");
        
        buildTestModel("resources/processed_data/data_train_clean_Cluster.arff",
                "resources/processed_data/data_test_clean_Cluster.arff",
                "resources/processed_data/sample_clean_Cluster.arff",
                "data_clean_Cluster");

        buildTestModel("resources/processed_data/data_train_Group_clean.arff",
                "resources/processed_data/data_test_Group_clean.arff",
                "resources/processed_data/sample_Group_clean.arff",
                "data_Group_clean");

        buildTestModel("resources/processed_data/data_train_Group_Out_clean.arff",
                "resources/processed_data/data_test_Group_Out_clean.arff",
                "resources/processed_data/sample_Group_Out_clean.arff",
                "data_Group_Out_clean");

        buildTestModel("resources/processed_data/data_train_Group_Out_Cluster_clean.arff",
                "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
                "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
                "data_Group_Out_Cluster_clean");
        
        // // Test
        // // data clean
        // testModel("resources/model/IBk_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/RandomForest_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/LinearRegression_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/DecisionTable_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/M5Rules_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/M5P_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");
        // testModel("resources/model/MultilayerPerceptron_data_clean_model.model",
        //         "resources/processed_data/data_test_clean.arff",
        //         "resources/processed_data/sample_clean.arff",
        //         "data clean");


        // // data clean cluster
        // testModel("resources/model/IBk_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/RandomForest_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/LinearRegression_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/DecisionTable_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/M5Rules_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/M5P_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");
        // testModel("resources/model/MultilayerPerceptron_data_clean_Cluster_model.model",
        //         "resources/processed_data/data_test_clean_Cluster.arff",
        //         "resources/processed_data/sample_clean_Cluster.arff",
        //         "data clean cluster");

        // // data group clean
        // testModel("resources/model/IBk_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/RandomForest_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/LinearRegression_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/DecisionTable_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/M5Rules_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/M5P_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");
        // testModel("resources/model/MultilayerPerceptron_data_Group_clean_model.model",
        //         "resources/processed_data/data_test_Group_clean.arff",
        //         "resources/processed_data/sample_Group_clean.arff",
        //         "data group clean");


        // // data group clean out
        // testModel("resources/model/IBk_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/RandomForest_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/LinearRegression_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/DecisionTable_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/M5Rules_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/M5P_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");
        // testModel("resources/model/MultilayerPerceptron_data_Group_Out_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_clean.arff",
        //         "resources/processed_data/sample_Group_Out_clean.arff",
        //         "data group clean remove outline");


        // // data group clean out cluster
        // testModel("resources/model/IBk_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/RandomForest_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/LinearRegression_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/DecisionTable_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/M5Rules_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/M5P_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");
        // testModel("resources/model/MultilayerPerceptron_data_Group_Out_Cluster_clean_model.model",
        //         "resources/processed_data/data_test_Group_Out_Cluster_clean.arff",
        //         "resources/processed_data/sample_Group_Out_Cluster_clean.arff",
        //         "data group clean remove outline cluster");

        // Step 5: Cross-validation for comparison
        runModelFold("resources/processed_data/data_train_clean.arff");
        runModelFold("resources/processed_data/data_train_clean_Cluster.arff");
        runModelFold("resources/processed_data/data_train_Group_clean.arff");
        runModelFold("resources/processed_data/data_train_Group_Out_clean.arff");
        runModelFold("resources/processed_data/data_train_Group_Out_Cluster_clean.arff");

        // Step 6: Predict using 10-fold trained models
        testModel("resources/model_10Fold/RandomForest_data_clean_model_10Fold.model",
                "resources/processed_data/data_test_clean.arff",
                "resources/processed_data/sample_clean.arff",
                "data clean");

        testModel("resources/model_10Fold/MultilayerPerceptron_data_Group_Out_clean_model_10Fold.model",
                "resources/processed_data/data_test_Group_Out_clean.arff",
                "resources/processed_data/sample_Group_Out_clean.arff",
                "data group clean remove outline");

        
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

    public static void buildTestModel(String trainPath,String testPath,String SamplePath,String file) throws Exception {
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
        Instances dataSample = loadData(SamplePath);

        System.out.println("Running file: " + file);

        System.out.println("Number of instances in trainData: " + dataTrain.size());
        System.out.println("Number of instances in testData: " + dataTest.size());
        System.out.println("Number of instances in sampleData: " + dataSample.size());

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
            System.out.println("Error Rate: " + eval.errorRate());

            System.out.println("=== Evaluating model on sample data ===");
            Evaluation evalSample = new Evaluation(dataSample);
            evalSample.evaluateModel(model, dataSample);
        
            System.out.println(evalSample.toSummaryString());
            System.out.println("Correlation coefficient: " + evalSample.correlationCoefficient());
            System.out.println("Root Mean Squared Error (RMSE): " + evalSample.rootMeanSquaredError());
            System.out.println("Mean Absolute Error (MAE): " + evalSample.meanAbsoluteError());
            System.out.println("Error Rate: " + evalSample.errorRate());

            String modelFilePath = "resources/model/" + model.getClass().getSimpleName() +"_"+ file + "_model.model";
            saveModel(model, modelFilePath);
        }

    }

    public static void testModel(String pathMode,String testPath,String SamplePath,String file) throws Exception {

        Classifier model = loadModel(pathMode);
        Instances dataTest = loadData(testPath);
        Instances dataSample = loadData(SamplePath);

        System.out.println("\nRunning file: " + file);

        System.out.println("Number of instances in testData: " + dataTest.size());
        System.out.println("Number of instances in sampleData: " + dataSample.size());

        System.out.println("\n=== Evaluating model: " + model.getClass().getSimpleName() + " ===");

        System.out.println("=== Evaluating model on test data ===");
        Evaluation eval = new Evaluation(dataTest);
        eval.evaluateModel(model, dataTest);
        
        System.out.println(eval.toSummaryString());
        System.out.println("Correlation coefficient: " + eval.correlationCoefficient());
        System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
        System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
        System.out.println("Error Rate: " + eval.errorRate());

        System.out.println("=== Evaluating model on sample data ===");
        Evaluation evalSample = new Evaluation(dataSample);
        evalSample.evaluateModel(model, dataSample);
        
        System.out.println(evalSample.toSummaryString());
        System.out.println("Correlation coefficient: " + evalSample.correlationCoefficient());
        System.out.println("Root Mean Squared Error (RMSE): " + evalSample.rootMeanSquaredError());
        System.out.println("Mean Absolute Error (MAE): " + evalSample.meanAbsoluteError());
        System.out.println("Error Rate: " + evalSample.errorRate());

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
        
            System.out.println("Time taken to train model: " + trainingDuration + " milliseconds");

            String modelFilePath = "resources/model/" + model.getClass().getSimpleName() +"_"+ file +"_model.model";
            saveModel(model, modelFilePath);
        }

    }
    

    public static void convertAllCsvToArff() {
        final String inputPathX = "resources/original_data/data_X.csv";
        final String inputPathY = "resources/original_data/data_Y.csv";
        final String inputSample = "resources/original_data/sample_submission.csv";

        final String outputArffX = "resources/processed_data/data_X.arff";
        saveDataXToArff(inputPathX, outputArffX);

        final String outputArffY = "resources/processed_data/data_Y.arff";
        saveDataYToArff(inputPathY, outputArffY);

        final String outputArffSample = "resources/processed_data/sample_submission.arff";
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
        data = removeOutliersTrain(data, model.outlierThresholds);

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

        data = removeOutliersTest(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        return data;
    }

    public static Instances preprocessTrainDataCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = removeOutliersTrain(data, model.outlierThresholds);

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

    public static Instances preprocessTestDataCluster(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = removeOutliersTest(data, model.outlierThresholds);

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
        data = removeOutliersTrain(data, model.outlierThresholds);

        return data;
    }

    public static Instances preprocessTestDataGroupRemoveOutliers(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = removeOutliersTest(data, model.outlierThresholds);

        return data;
    }

    public static Instances preprocessTrainDataGroupRemoveOutliersNorminal(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        model.outlierThresholds = calculateOutlierThresholds(data);
        data = removeOutliersTrain(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

        return data;
    }

    public static Instances preprocessTestDataGroupRemoveOutliersNorminal(Instances data, PreprocessingModel model) throws Exception {
        ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
        replaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissingValues);

        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        removeDuplicates.setInputFormat(data);
        data = Filter.useFilter(data, removeDuplicates);

        data = removeOutliersTest(data, model.outlierThresholds);

        model.normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, model.normalizeFilter);

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

    public static Instances removeOutliersTrain(Instances data, OutlierThresholds thresholds) {
        int classIndex = data.classIndex();

        for (int i = data.numInstances() - 1; i >= 0; i--) {
            Instance instance = data.instance(i);
            for (int attrIndex : thresholds.bounds.keySet()) {
                if (attrIndex == classIndex) continue;
                double val = instance.value(attrIndex);
                double[] bound = thresholds.bounds.get(attrIndex);
                if (val < bound[0] || val > bound[1]) {
                    data.delete(i);
                    break;
                }
            }
        }

        data.setClassIndex(classIndex);
        return data;
    }

    public static Instances removeOutliersTest(Instances data, OutlierThresholds thresholds) {
        int classIndex = data.classIndex();

        for (int i = data.numInstances() - 1; i >= 0; i--) {
            Instance instance = data.instance(i);
            for (int attrIndex : thresholds.bounds.keySet()) {
                if (attrIndex == classIndex) continue;
                double val = instance.value(attrIndex);
                double[] bound = thresholds.bounds.get(attrIndex);
                if (val < bound[0] || val > bound[1]) {
                    data.delete(i);
                    break;
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

    public static void saveModel(Classifier model, String modelFilePath) throws Exception {
        SerializationHelper.write(modelFilePath, model);
        System.out.println("Model saved to: " + modelFilePath);
    }

    public static void processFiles() throws Exception {
        String dataXPath = "resources/original_data/data_X.csv";
        String dataYPath = "resources/original_data/data_Y.csv";
        String samplePath = "resources/original_data/sample_submission.csv";

        String[][] dataX = readData(dataXPath);
        String[][] dataY = readData(dataYPath);
        String[][] sample = readData(samplePath);

        String outputArffPath = "resources/processed_data/data.arff";

        BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputArffPath));
        writeArffHeader(writer);

        writeData(writer,dataX, dataY);

        writer.close();

        String outputArffPathSample = "resources/processed_data/data_Sample.arff";

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

        // writer.write("@ATTRIBUTE data_time DATE \"yyyy-MM-dd'T'HH:mm:ss\"\n");  // Cột thời gian với 'T'
    
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
            System.out.println("Lưu dữ liệu dataY hoàn tất: " + outputArffY);
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
            System.out.println("Lưu dữ liệu dataX hoàn tất: " + outputArffX);
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


    public static void processFileCombine() throws Exception {
        String dataXPath = "resources/original_data/data_X.csv";
        String dataYPath = "resources/original_data/data_Y.csv";
        String samplePath = "resources/original_data/sample_submission.csv";

        String[][] dataX = readData(dataXPath);
        String[][] dataY = readData(dataYPath);
        String[][] sample = readData(samplePath);

        String outputArffPath = "resources/processed_data/dataCombine.arff";

        BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputArffPath));
        writeArffHeaderCombine(writer);

        writeDataCombine(writer,dataX, dataY);

        writer.close();

        String outputArffPathSample = "resources/processed_data/dataCombine_Sample.arff";

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

    
}