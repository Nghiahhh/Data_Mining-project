package data_mining.project;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;


public class ModelFactory {

    public static Classifier getLinearRegressionModel() {
        LinearRegression model = new LinearRegression();
        model.setEliminateColinearAttributes(true);
        return model;
    }


    public static Classifier getDecisionTableModel() {
        DecisionTable model = new DecisionTable();
        model.setUseIBk(true);
        return model;
    }

    public static Classifier getM5RulesModel() {
        M5Rules model = new M5Rules();
        model.setMinNumInstances(5);
        return model;
    }

    public static Classifier getIBkModel() {
        IBk model = new IBk();
        model.setKNN(3);
        model.setCrossValidate(false);
        return model;
    }

    public static Classifier getRandomForestModel() {
        RandomForest model = new RandomForest();
        model.setNumIterations(100);
        model.setNumFeatures(0); 
        model.setMaxDepth(10);
        return model;
    }

    public static Classifier getM5PModel() {
        M5P model = new M5P();
        model.setMinNumInstances(4);
        return model;
    }


    public static Classifier getMultilayerPerceptronModel() {
        MultilayerPerceptron model = new MultilayerPerceptron();
        model.setLearningRate(0.1);
        model.setMomentum(0.2);
        model.setTrainingTime(500);  
        model.setHiddenLayers("a");  
        return model;
    }
}
