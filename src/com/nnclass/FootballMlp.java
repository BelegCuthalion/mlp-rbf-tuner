package com.nnclass;

import com.nnclass.nadashteha.Helper;
import com.nnclass.nadashteha.OnlineGrapher;
import com.nnclass.nadashteha.PercentConfusionMatrix;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
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
 * Created by beleg on 11/25/16.
 */
public class FootballMlp {
    private double theBestAccuracy = 0;
    private double[] theMostAccurate;

    private static FootballMlp ourInstance = new FootballMlp();

    public static FootballMlp getInstance() {
        return ourInstance;
    }

    private FootballMlp() {
    }

    public double[] getTheMostAccurate() {
        return theMostAccurate;
    }

    public void run(double learningRate, double momentum, int iteration, int...topology) {
        DataSet rawSet;
        DataSet[] dataSet = new DataSet[2];
        try {
//            rawSet = TrainingSetImport.importFromFile("PremierLeagueResults.txt", 8, 3, "\t");
            rawSet = TrainingSetImport.importFromFile("AnimalSpecies.txt", 16, 7, ",");
            rawSet.shuffle();
            dataSet = rawSet.createTrainingAndTestSubsets(30, 70);
        } catch (IOException e) {
            System.out.println("File error!");
        }
        DataSet trainSet = dataSet[0], testSet = dataSet[1];
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(topology);
        mlp.randomizeWeights(-0.5, 0.5);
        final MomentumBackpropagation lesson = new MomentumBackpropagation();
        lesson.setLearningRate(learningRate);
        lesson.setMomentum(momentum);
        lesson.setMaxError(0.001);
        lesson.setMaxIterations(50000);
        lesson.setMinErrorChange(10e-32);
        lesson.setErrorFunction(new MeanSquaredError());
        lesson.setBatchMode(true);
        EpochEndEventListener listener = new EpochEndEventListener(lesson);
        lesson.addListener(listener);

        mlp.learn(trainSet, lesson);

        double[][] errorGraph = new double[2][listener.errorGraph.size()];
        for(int i=0; i<listener.errorGraph.size(); i++) {
            errorGraph[0][i] = i;
            errorGraph[1][i] = (Double)listener.errorGraph.toArray()[i];
        }

        OnlineGrapher.getInstance().init(errorGraph, "Train error graph", "Epoch", "MSE", "mse");
        OnlineGrapher.getInstance().save("results/" + iteration);

//        PercentConfusionMatrix trainConfusionMatrix = new PercentConfusionMatrix(new String[]{"Win", "Draw", "Lose"}, 3);
//        PercentConfusionMatrix testConfusionMatrix = new PercentConfusionMatrix(new String[]{"Win", "Draw", "Lose"}, 3);

        PercentConfusionMatrix trainConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);
        PercentConfusionMatrix testConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);

        for(DataSetRow sample : trainSet.getRows()) {
            mlp.setInput(sample.getInput());
            mlp.calculate();
            int selectedClass = Helper.argMax(mlp.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            trainConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        for(DataSetRow sample : testSet.getRows()) {
            mlp.setInput(sample.getInput());
            mlp.calculate();
            int selectedClass = Helper.argMax(mlp.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            testConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        testConfusionMatrix.computePercentValues();
        trainConfusionMatrix.computePercentValues();

        StringBuilder topologyStringBuilder = new StringBuilder();
        int totalHiddenNeurons = 0;
        for(int n : topology) {
            topologyStringBuilder.append("-" + n);
            totalHiddenNeurons += n;
        }
        totalHiddenNeurons -= mlp.getInputsCount();
        totalHiddenNeurons -= mlp.getOutputsCount();
        double apn = (100d*testConfusionMatrix.getPercentValueAt(testConfusionMatrix.getClassCount(), testConfusionMatrix.getClassCount()))/(double)totalHiddenNeurons;
        List<String> fileOutput = new ArrayList<String>();
        fileOutput.add("ATTEMPT#" + iteration + ":");
        fileOutput.add("Topology: ");
        fileOutput.add(topologyStringBuilder.toString());
        fileOutput.add("Total Hidden Neurons: " + totalHiddenNeurons);
        fileOutput.add("Learning Rate: " + learningRate);
        fileOutput.add("Momentum: " + momentum);
        fileOutput.add("accuracy/neuron: " + apn);
        fileOutput.add("Train Results:");
        fileOutput.add(trainConfusionMatrix.rawTable());
        fileOutput.add(trainConfusionMatrix.toString());

        fileOutput.add("Test Results:");
        fileOutput.add(testConfusionMatrix.rawTable());
        fileOutput.add(testConfusionMatrix.toString());
        fileOutput.add("lastErr: " + listener.lastError + " #: " + listener.lastIteration + "\n");

        //  Elitism
        double[] eliteWeights = new double[listener.eliteWeights.length];
        for(int i=0; i<listener.eliteWeights.length; i++)
            eliteWeights[i] = listener.eliteWeights[i];

        mlp.setWeights(eliteWeights);

//        PercentConfusionMatrix eliteTrainConfusionMatrix = new PercentConfusionMatrix(new String[]{"Win", "Draw", "Lose"}, 3);
//        PercentConfusionMatrix eliteTestConfusionMatrix = new PercentConfusionMatrix(new String[]{"Win", "Draw", "Lose"}, 3);

        PercentConfusionMatrix eliteTrainConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);
        PercentConfusionMatrix eliteTestConfusionMatrix = new PercentConfusionMatrix(new String[]{"1", "2", "3", "4", "5", "6", "7"}, 7);

        for(DataSetRow sample : trainSet.getRows()) {
            mlp.setInput(sample.getInput());
            mlp.calculate();
            int selectedClass = Helper.argMax(mlp.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            eliteTrainConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        for(DataSetRow sample : testSet.getRows()) {
            mlp.setInput(sample.getInput());
            mlp.calculate();
            int selectedClass = Helper.argMax(mlp.getOutput());
            int actualClass = Helper.argMax(sample.getDesiredOutput());
            eliteTestConfusionMatrix.incrementElement(actualClass, selectedClass);
        }

        eliteTestConfusionMatrix.computePercentValues();
        eliteTrainConfusionMatrix.computePercentValues();

        fileOutput.add("Elite Train Results:");
        fileOutput.add(eliteTrainConfusionMatrix.rawTable());
        fileOutput.add(eliteTrainConfusionMatrix.toString());

        fileOutput.add("Elite Test Results:");
        fileOutput.add(eliteTestConfusionMatrix.rawTable());
        fileOutput.add(eliteTestConfusionMatrix.toString());
        fileOutput.add("minErr: " + listener.minError + " #: " + listener.minIteration);
        fileOutput.add("================================================================");

        List<String> rawFileOutput = new ArrayList<String>();
        rawFileOutput.add(iteration + "\t" + topologyStringBuilder.toString() + "\t" + learningRate + "\t" + momentum + "\t" + listener.lastError + "\t" + testConfusionMatrix.getPercentValueAt(topology[topology.length - 1], topology[topology.length - 1]) + "\t" + listener.minError + "\t" + eliteTestConfusionMatrix.getPercentValueAt(topology[topology.length-1], topology[topology.length-1]) + "\t" + apn);
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

        if(theBestAccuracy < eliteTestConfusionMatrix.getPercentValueAt(topology[topology.length-1], topology[topology.length-1])) {
            theBestAccuracy = eliteTestConfusionMatrix.getPercentValueAt(topology[topology.length - 1], topology[topology.length - 1]);
            theMostAccurate = eliteWeights;
        }

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
                this.lastError = ((MomentumBackpropagation) event.getSource()).getTotalNetworkError();
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
