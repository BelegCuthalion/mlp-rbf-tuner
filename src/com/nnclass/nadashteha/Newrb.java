package com.nnclass.nadashteha;

import Jama.Matrix;
import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.input.Difference;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.transfer.Gaussian;
import org.neuroph.nnet.learning.kmeans.Cluster;
import org.neuroph.nnet.learning.kmeans.KMeansClustering;
import org.neuroph.nnet.learning.kmeans.KVector;
import org.neuroph.util.ConnectionFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by beleg on 11/24/16.
 */
public class Newrb extends IterativeLearning {
//    private static double SPREAD = 0.1;
    double sigma, minError;
    Matrix targets;
    List<KVector> centroids;
    private double totalError = 1;
    private int maxRbfNeurons;

    public Newrb(double sigma, double minError, int maxRbfNeurons) {
        this.sigma = sigma;
        this.minError = minError;
        this.maxRbfNeurons = maxRbfNeurons;
    }

    protected Matrix rbfLayerOutput(DataSet dataSet, Layer rbfLayer, NeuralNetwork nn) {
        Matrix rbfOutputMatrix = new Matrix(dataSet.size(), rbfLayer.getNeuronsCount());
        int i=0, j;
        for(DataSetRow sample : dataSet.getRows()) {
            nn.setInput(sample.getInput());
            nn.calculate();
            j=0;
            for(Neuron rbfNeuron : rbfLayer.getNeurons()) {
                rbfOutputMatrix.set(i ,j, rbfNeuron.getOutput());
                j++;
            }
            i++;
        }
        return rbfOutputMatrix;
    }

    protected Matrix pseudoInverse(Matrix A) {
        return A.transpose().times(A).inverse().times(A.transpose());
    }

    public double getTotalNetworkError() {
        return this.totalError;
    }

    @Override
    public void onStart(){
        super.onStart();
        this.centroids = new ArrayList<>();
        this.targets = new Matrix(this.getTrainingSet().size(), this.getTrainingSet().getOutputSize());
        int i=0;
        for(DataSetRow sample : this.getTrainingSet().getRows()) {
            for(int j=0; j<this.getTrainingSet().getOutputSize(); j++) {
                this.targets.set(i, j, sample.getDesiredOutput()[j]);
            }
            i++;
        }

/*        for(int o=0; o<getTrainingSet().size(); o++){
            for(int p=0; p<this.neuralNetwork.getInputsCount(); p++) {
                this.neuralNetwork.getLayerAt(1).getNeurons()[o].getInputConnections()[p].getWeight().setValue(getTrainingSet().getRowAt(o).getInput()[p]);
            }
        }*/

        // set weights between input and rbf layer using kmeans
        KMeansClustering kmeans = new KMeansClustering(getTrainingSet());
        kmeans.setNumberOfClusters(neuralNetwork.getLayerAt(1).getNeuronsCount()); // set number of clusters as number of rbf neurons
        kmeans.doClustering();

        // get clusters (centroids)
        Cluster[] clusters = kmeans.getClusters();

        // assign each rbf neuron to one cluster
        // and use centroid vectors to initialize neuron's input weights
        Layer rbfLayer = neuralNetwork.getLayerAt(1);
        int k=0;
        for(Neuron neuron : rbfLayer.getNeurons()) {
            this.centroids.add(clusters[k].getCentroid());
            double[] weightValues = centroids.get(k).getValues();
            int c=0;
            for(Connection conn : neuron.getInputConnections()) {
                conn.getWeight().setValue(weightValues[c]);
                c++;
            }
            k++;
        }

/*
        // get cluster centroids as list
        List<KVector> centroids = new ArrayList<>();
        for(Cluster cluster : clusters) {
            centroids.add(cluster.getCentroid());
        }
*/

    updateSpreads();
    }

    public double getSigma() {
        return this.sigma;
    }

    private void updateSpreads() {
        // use KNN to calculate sigma param - gausssian function width for each neuron
//        KNearestNeighbour knn = new KNearestNeighbour();
//        knn.setDataSet(this.centroids);
//        int n = 0;
//        for(KVector centroid : this.centroids) {
////             calculate and set sigma for each neuron in rbf layer
//            KVector[] nearestNeighbours = knn.getKNearestNeighbours(centroid, 2);
//            double sigma = calculateSigma(centroid, nearestNeighbours); // calculate in method
//            Neuron neuron = this.neuralNetwork.getLayerAt(1).getNeuronAt(n);
//            ((Gaussian)neuron.getTransferFunction()).setSigma(SPREAD);
//            n++;
//        }
        for(Neuron neuron : this.neuralNetwork.getLayerAt(1).getNeurons()) {
            ((Gaussian)neuron.getTransferFunction()).setSigma(sigma);
        }
    }

    @Override
    public void doLearningEpoch(DataSet trainingSet) {
        Matrix rbfLayerOutput = rbfLayerOutput(trainingSet, this.neuralNetwork.getLayerAt(1), this.neuralNetwork);
        Matrix mPlus = Pinv.pinv(rbfLayerOutput);
        Matrix weights = null;
        if(mPlus != null)
            weights = mPlus.times(this.targets);
        else {
            stopLearning();
            return;
        }
        for(int i=0; i<this.neuralNetwork.getLayerAt(1).getNeuronsCount(); i++)
            for(int j=0; j<this.neuralNetwork.getOutputsCount(); j++)
                this.neuralNetwork.getLayerAt(1).getNeurons()[i].getOutConnections()[j].setWeight(new Weight(weights.get(i, j)));

        if(this.neuralNetwork.getLayerAt(1).getNeuronsCount() <= trainingSet.size() && this.neuralNetwork.getLayerAt(1).getNeuronsCount() <= this.maxRbfNeurons && this.totalError > minError){
            DataSetRow maxErrorSample = null;
            double maxError = 0, totalError = 0;
            for(DataSetRow sample : trainingSet.getRows()){
                this.neuralNetwork.setInput(sample.getInput());
                this.neuralNetwork.calculate();
                KVector output = new KVector(this.neuralNetwork.getOutput());
                KVector target = new KVector(sample.getDesiredOutput());
                double sampleError = output.distanceFrom(target);
                if(sampleError > maxError) {
                    maxError = sampleError;
                    maxErrorSample = sample;
                }
                totalError += sampleError * sampleError;
            }
            totalError = totalError/(trainingSet.size());
            Neuron newNeuron = new Neuron(new Difference(), new Gaussian());
            this.neuralNetwork.getLayerAt(1).addNeuron(newNeuron);
            for(int k=0; k<this.neuralNetwork.getInputsCount(); k++)
                ConnectionFactory.createConnection(this.neuralNetwork.getInputNeurons()[k], newNeuron, maxErrorSample.getInput()[k]);
            ConnectionFactory.createConnection(newNeuron, this.neuralNetwork.getLayerAt(2));
            this.centroids.add(new KVector(maxErrorSample.getInput()));
            ((Gaussian)newNeuron.getTransferFunction()).setSigma(sigma);
            this.totalError = totalError;
        } else {
            stopLearning();
        }
    }

    private double calculateSigma(KVector centroid,  KVector[] nearestNeighbours) {
        double sigma = 0;

        for(KVector nn : nearestNeighbours){
            sigma += Math.pow( centroid.distanceFrom(nn), 2 ) ;
        }

        sigma = Math.sqrt(1/((double)nearestNeighbours.length)  * sigma);

        return sigma;
    }

}
