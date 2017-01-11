package com.nnclass.nadashteha;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by beleg on 11/25/16.
 */
public class OnlineGrapher {
    SwingWrapper<XYChart> swingWrapper;
    XYChart chart;
    List<Double> xData;
    List<Double> yData;
    private static OnlineGrapher ourInstance = new OnlineGrapher();

    public static OnlineGrapher getInstance() {
        return ourInstance;
    }

    private OnlineGrapher() {
    }

    public void init(double [][] initData, String windowName, String xName, String yName, String seriesName) {
//        this.xData = Arrays.asList(initData[0]);
//        this.yData = Arrays.asList(initData[1]);
        // Create Chart
        this.chart = QuickChart.getChart(windowName, xName, yName, seriesName, initData[0], initData[1]);

        // Show it
        this.swingWrapper = new SwingWrapper<XYChart>(chart);
    }

    public void show() {
        this.swingWrapper.displayChart();
    }

    public void save(String fileName) {
        try {
            BitmapEncoder.saveBitmap(chart, fileName, BitmapEncoder.BitmapFormat.JPG);
        } catch (IOException e) {
            System.out.println("Can't save graph!");
            e.printStackTrace();
        }
    }

    public void update(final String seriesName, final double x, final double y) {
        final double[] xs = new double[1];
        final double[] ys = new double[1];
        xs[0] = x;
        ys[0] = y;
        javax.swing.SwingUtilities.invokeLater(new Runnable() {

            @Override
            public void run() {

                chart.updateXYSeries(seriesName, xs, ys, null);
                swingWrapper.repaintChart();
            }
        });
    }

    public void add(double x, double y) {
            this.xData.add(x);
            this.yData.add(y);
    }

    public void update(final String seriesName, final double[][] data) {
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                chart.updateXYSeries(seriesName, data[0], data[1], null);
                swingWrapper.repaintChart();
            }
        });
    }
}
