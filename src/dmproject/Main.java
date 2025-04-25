/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package dmproject;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load dữ liệu từ file ARFF
        DataSource source = new DataSource("src/dmproject/resources/segment-challenge.arff");
        Instances data = source.getDataSet();

        // Thiết lập cột class (label) là cột cuối cùng
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Sử dụng thuật toán J48 (Decision Tree)
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Đánh giá bằng 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        // In kết quả
        System.out.println(eval.toSummaryString("\n=== Evaluation Result ===\n", false));
    }
}

