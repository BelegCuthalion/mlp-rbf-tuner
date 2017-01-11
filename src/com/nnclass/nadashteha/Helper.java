package com.nnclass.nadashteha;

/**
 * Created by beleg on 11/25/16.
 */
public final class Helper {
    public static int argMax(double...params) {
        double max = 0;
        int maxIdx = 0;
        for(int i=0; i<params.length; i++)
            if(params[i] > max) {
                max = params[i];
                maxIdx = i;
            }
        return maxIdx;
    }
}
