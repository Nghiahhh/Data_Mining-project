package main.data_mining.project;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;

/**
 * Lớp tạo các mô hình hồi quy (regression models) sử dụng thư viện WEKA.
 */
public class ModelFactory {

    /**
     * Linear Regression - hồi quy tuyến tính
     */
    public static Classifier getLinearRegressionModel() {
        LinearRegression model = new LinearRegression();
        model.setEliminateColinearAttributes(true); // Loại bỏ thuộc tính đồng tuyến tính
        return model;
    }

    /**
     * Decision Table - bảng luật, dùng IBk làm classifier phụ
     */
    public static Classifier getDecisionTableModel() {
        DecisionTable model = new DecisionTable();
        model.setUseIBk(true); // Dùng IBk thay vì ZeroR
        return model;
    }

    /**
     * M5Rules - mô hình cây kết hợp luật
     */
    public static Classifier getM5RulesModel() {
        M5Rules model = new M5Rules();
        model.setMinNumInstances(4); // Số lượng nhỏ nhất của instance để phân tách
        return model;
    }

    /**
     * IBk - K-Nearest Neighbors
     */
    public static Classifier getIBkModel() {
        IBk model = new IBk();
        model.setKNN(3); // Số lượng láng giềng gần nhất
        model.setCrossValidate(false); // Không dùng CV để chọn K
        return model;
    }

    /**
     * Random Forest - rừng ngẫu nhiên (sử dụng cho regression)
     */
    public static Classifier getRandomForestModel() {
        RandomForest model = new RandomForest();
        model.setNumIterations(100); // Số cây trong rừng
        model.setNumFeatures(0);     // Dùng sqrt(#features) nếu = 0
        model.setMaxDepth(10);       // Giới hạn độ sâu
        return model;
    }

    /**
     * M5P - cây hồi quy M5P
     */
    public static Classifier getM5PModel() {
        M5P model = new M5P();
        model.setMinNumInstances(4);
        return model;
    }

    /**
     * Multilayer Perceptron - mạng nơ-ron hồi quy
     */
    public static Classifier getMultilayerPerceptronModel() {
        MultilayerPerceptron model = new MultilayerPerceptron();
        model.setLearningRate(0.1);
        model.setMomentum(0.2);
        model.setTrainingTime(500);     // Số epoch
        model.setHiddenLayers("a");     // (attribs + classes) / 2
        return model;
    }
}

