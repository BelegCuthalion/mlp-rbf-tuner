package com.nnclass;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

public class Main {
    public static void main(String[] args) {
        double bestSetting = 0;
        int i = 1381;
//        for(double error=0; error<1; error+=0.05)
//            for(double sigma=0.008; sigma<2; sigma+=0.001)
//                for(int j=0; j<5; j++){
//                    FootballRbf.getInstance().run(i, sigma, error, 80);
//                    i++;
//                }
//        for(double eta=0.1; eta<0.6; eta+=0.1)
//            for(double moment=0.7; moment<0.9; moment+=0.1)
//                for(int h=2; h<20; h++) {
//                    FootballMlp.getInstance().run(eta, moment, i, 16, h, 7);
//                    i++;
//                }
//
//        for(double eta=0.01; eta<0.6; eta+=0.1)
//            for(double moment=0.6; moment<0.9; moment+=0.1)
//                for(int h=2; h<100; h++) {
//                    FootballMlp.getInstance().run(eta, moment, i, 16, 8, h, 7);
//                    i++;
//                }
//        for(int j=0; j<4; j++) {
//            FootballMlp.getInstance().run(0.01, 0.6, i, 16, 18, 7);
//            i++;
//        }
//
//        for(int j=0; j<5; j++) {
//            FootballMlp.getInstance().run(0.01, 0.7, i, 16, 18, 7);
//            i++;
//        }
//
//        for(int j=0; j<5; j++) {
//            FootballMlp.getInstance().run(0.01, 0.8, i, 16, 18, 7);
//            i++;
//        }

        for(double minErr=0.02; minErr<0.1; minErr+=0.01)
            for(double spread=0.01; spread<1; spread+=0.01)
                for(int j=0; j<4; j++) {
                    if(!(spread==0.03 && minErr==0.02))
                        FootballRbf.getInstance().run(i, spread, minErr);
                    i++;
                }
    }
}