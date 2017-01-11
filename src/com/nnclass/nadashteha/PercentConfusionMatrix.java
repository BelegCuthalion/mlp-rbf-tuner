package com.nnclass.nadashteha;

import org.neuroph.contrib.eval.classification.ConfusionMatrix;

import java.math.RoundingMode;
import java.text.DecimalFormat;

/**
 * Created by beleg on 11/25/16.
 */
public class PercentConfusionMatrix extends ConfusionMatrix {
//    private double[][] values;
    private double[][] percentValues;
    private int classCount;
    private String[] classLabels;

    public PercentConfusionMatrix(String[] labels, int classCount) {
        super(labels, classCount);
        this.classCount = classCount;
        this.classLabels = labels;
        this.percentValues = new double[classCount + 1][classCount + 1];
    }

    public int getClassCount() {
        return classCount;
    }

    public double getPercentValueAt(int i, int j) {
        return percentValues[i][j];
    }

    public void computePercentValues() {
        for(int i=0; i<classCount; i++)
            for(int j=0; j<classCount; j++)
                percentValues[i][j] = getValueAt(i, j)/getTotal();
        for(int i=0; i<classCount; i++) {
            int classTotal = 0;
            for(int j=0; j<classCount; j++)
                classTotal += getValueAt(i, j);
            percentValues[i][classCount] = getValueAt(i, i)/classTotal;
        }
        for(int j=0; j<classCount; j++) {
            int predictedTotal = 0;
            for(int i=0; i<classCount; i++)
                predictedTotal += getValueAt(i, j);
            percentValues[classCount][j] = getValueAt(j, j)/predictedTotal;
        }
        int trueClassified = 0;
        for(int i=0; i<classCount; i++)
            trueClassified += getValueAt(i, i);
        percentValues[classCount][classCount] = trueClassified/(double)getTotal();
    }

    public String rawTable() {
        return super.toString();
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        DecimalFormat df = new DecimalFormat("#.##");
        df.setRoundingMode(RoundingMode.CEILING);

        int maxColumnLenght = 7;
        for (String label : classLabels)
            maxColumnLenght = Math.max(maxColumnLenght, label.length()) + 1;

        builder.append(String.format("%1$" + maxColumnLenght + "s", ""));
        for (String label : classLabels)
            builder.append(String.format("%1$" + maxColumnLenght + "s", label));
        builder.append("\n");

        for (int i = 0; i < percentValues.length; i++) {
            if(i < classCount)
                builder.append(String.format("%1$" + maxColumnLenght + "s", classLabels[i]));
            else
                builder.append(String.format("%1$" + maxColumnLenght + "s", ""));
            for (int j = 0; j < percentValues[0].length; j++) {
                builder.append(String.format("%1$" + maxColumnLenght + "s", df.format(percentValues[i][j])));
            }
            builder.append("\n");

        }
        return builder.toString();
    }

}
