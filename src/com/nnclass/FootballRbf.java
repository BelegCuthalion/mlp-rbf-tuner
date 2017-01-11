package com.nnclass;

import com.nnclass.nadashteha.*;
import org.neuroph.contrib.eval.ClassificationEvaluator;
import org.neuroph.contrib.eval.classification.ConfusionMatrix;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.RBFNetwork;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.nnet.learning.RBFLearning;
import org.neuroph.util.TrainingSetImport;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by beleg on 11/23/16.
 */
public class FootballRbf {
    private static FootballRbf ourInstance = new FootballRbf();

    public static FootballRbf getInstance() {
        return ourInstance;
    }

    private FootballRbf() {
    }

    public double run(int iteration, double spread, double minError) {
        return this.run(iteration, spread, minError, (int)10e+4);
    }

    public double run(int iteration, double spread, double minError, int maxRbfNeurons){
        DataSet[] dataSet = new DataSet[2];
        try {
//            dataSet = TrainingSetImport.importFromFile("PremierLeagueResults.txt", 8, 3, "\t").createTrainingAndTestSubsets(70, 30);
//            dataSet = TrainingSetImport.importFromFile("test.txt", 2, 1, ",").createTrainingAndTestSubsets(100, 0);
            dataSet = TrainingSetImport.importFromFile("AnimalSpecies.txt", 16, 7, ",").createTrainingAndTestSubsets(30, 70);
        } catch (IOException e) {
            System.out.println("File error!");
        }

        DataSet trainSet = dataSet[0], testSet = dataSet[1];
        RBFNetwork rbf = new RBFNetwork(16, 2, 7);
//        rbf.randomizeWeights(-0.5, 0.5);
        Newrb lesson = new Newrb(spread, minError, maxRbfNeurons);
//        final RBFBack lesson = new RBFBack();
//        final RBFLearning lesson = new RBFLearning();
//        lesson.addListener(new LearningEventListener() {
//            @Override
//            public void handleLearningEvent(LearningEvent event) {
//                if (event.getEventType() == LearningEvent.Type.EPOCH_ENDED) {
//                    System.out.println("TOTAL#ERROR: " + ((LMS) event.getSource()).getTotalNetworkError());
//                    if (lesson.getTotalNetworkError() < 0.3 || lesson.getCurrentIteration() > 10E+5)
//                        lesson.stopLearning();
//                }
//            }
//        });
        EpochEndEventListener listener = new EpochEndEventListener(lesson);
        lesson.addListener(listener);

        rbf.learn(trainSet, lesson);

        double[][] errorGraph = new double[2][listener.errorGraph.size()];
        for(int i=0; i<listener.errorGraph.size(); i++) {
            errorGraph[0][i] = i;
            errorGraph[1][i] = (Double)listener.errorGraph.toArray()[i];
        }

        OnlineGrapher.getInstance().init(errorGraph, "Train error graph", "RBF Neuron", "MSE", "mse");
        OnlineGrapher.getInstance().save("results/" + iteration);

        PercentConfusionMatrix trainConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);
        PercentConfusionMatrix testConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);

        for(DataSetRow sample : trainSet.getRows()) {
            rbf.setInput(sample.getInput());
            rbf.calculate();
            int selectedClass = Helper.argMax(rbf.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            trainConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        for(DataSetRow sample : testSet.getRows()) {
            rbf.setInput(sample.getInput());
            rbf.calculate();
            int selectedClass = Helper.argMax(rbf.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            testConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        trainConfusionMatrix.computePercentValues();
        testConfusionMatrix.computePercentValues();

        double apn = (100d*testConfusionMatrix.getPercentValueAt(testConfusionMatrix.getClassCount(), testConfusionMatrix.getClassCount()))/((double)rbf.getLayerAt(1).getNeuronsCount());

        List<String> fileOutput = new ArrayList<String>();
        fileOutput.add("ATTEMPT#" + iteration + ":");
        fileOutput.add("RBF Neurons: " + rbf.getLayerAt(1).getNeuronsCount());
        fileOutput.add("sigma: " + lesson.getSigma());
        fileOutput.add("minError: " + minError);
        fileOutput.add("accuracy/neuron: " + apn);
        fileOutput.add("Train Results:");
        fileOutput.add(trainConfusionMatrix.rawTable());
        fileOutput.add(trainConfusionMatrix.toString());

        fileOutput.add("Test Results:");
        fileOutput.add(testConfusionMatrix.rawTable());
        fileOutput.add(testConfusionMatrix.toString());
        fileOutput.add("lastErr: " + listener.lastError + " #: " + (listener.lastIteration+3) + "\n");
        fileOutput.add("================================================================");
        List<String> rawFileOutput = new ArrayList<String>();
        rawFileOutput.add(iteration + "\t" + rbf.getLayerAt(1).getNeuronsCount() + "\t" + minError + "\t" + spread + "\t" + listener.lastError + "\t" + testConfusionMatrix.getPercentValueAt(testConfusionMatrix.getClassCount(), testConfusionMatrix.getClassCount()) + "\t" + apn);
        System.out.println(rawFileOutput.get(rawFileOutput.size()-1));
        Path outputFile = Paths.get("results/results.txt");
        Path rawFile = Paths.get("results/results.raw");
        try {
            Files.write(outputFile, fileOutput, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
            Files.write(rawFile, rawFileOutput, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can't open output file!");
        }
        return testConfusionMatrix.getPercentValueAt(testConfusionMatrix.getClassCount(), testConfusionMatrix.getClassCount());
    }

    class EpochEndEventListener implements LearningEventListener {
        double minError = 10E+10;
        double lastError = 0;
        IterativeLearning lesson;
        Double[] eliteWeights;
        int minIteration;
        int lastIteration;
        List<Double> errorGraph = new ArrayList<>();

        public EpochEndEventListener(IterativeLearning lesson) {
            this.lesson = lesson;
        }
        @Override
        public void handleLearningEvent(LearningEvent event) {
            if (event.getEventType() == LearningEvent.Type.EPOCH_ENDED) {
                this.lastError = ((Newrb) event.getSource()).getTotalNetworkError();
                this.errorGraph.add(lastError);
                this.lastIteration = this.lesson.getCurrentIteration();
                if(this.minError > this.lastError) {
                    this.minError = this.lastError;
                    this.minIteration = this.lastIteration;
                    this.eliteWeights = this.lesson.getNeuralNetwork().getWeights();
                }
            }
        }
    }
}
