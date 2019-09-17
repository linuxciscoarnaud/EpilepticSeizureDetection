/**
 * 
 */
package com.epilepticseizuredetection;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * @author Arnaud
 *
 */

public class Params {

	// Parameters for network configuration
		private OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
		private Activation activation = Activation.IDENTITY;
		private WeightInit weightInit = WeightInit.XAVIER; 
		private IUpdater updater = new AdaDelta(); 
		private CacheMode cacheMode = CacheMode.NONE;
		private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
		private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
		
		// Parameters for input data
		private int height = 224;
		private int width = 224;
		private int channels = 3; // We are dealing with gray scale images
	    
	    // Parameters for the training phase
	    protected static int miniBatchSize = 1;
	    protected static int epochs = 200;
	    protected static int maxTimeIterTerminationCondition = 7; // 7 hours
	    
	    protected static long seed = 123; // Integer for reproducibility of a random number generator
	    protected static Random rng = new Random(seed);
	    
	    // Getters
		
	 	public OptimizationAlgorithm getOptimizationAlgorithm() {
	 		return optimizationAlgorithm;
	 	}
	 	
	 	public Activation getActivation() {
	 		return activation;
	 	}
	 	
	 	public WeightInit getWeightInit() {
	 		return weightInit;
	 	}
	 	
	 	public IUpdater getUpdater() {
	 		return updater;
	 	}
	 	
	 	public CacheMode getCacheMode() {
	 		return cacheMode;
	 	}
	 	
	 	public WorkspaceMode getWorkspaceMode() {
	 		return workspaceMode;
	 	}
	 	
	 	public ConvolutionLayer.AlgoMode getCudnnAlgoMode() {
	 		return cudnnAlgoMode;
	 	}
	 	
	 	public int getEpochs() {
	 		return epochs;
	 	}
	 	
	 	public int getMaxTimeIterTerminationCondition() {
	 		return maxTimeIterTerminationCondition;
	 	}
	 	
	 	public int getMiniBatchSize() {
	 		return miniBatchSize;
	 	}
	 	
	 	public int getHeight() {
	 		return height;
	 	}
	 	
	 	public int getWidth( ) {
	 		return width;
	 	}
	 	
	 	public int getChannels() {
	 		return channels;
	 	}
	 	
	 	public long getSeed() {
	 		return seed;
	 	}
	 	
	 	public Random getRng() {
	 		return rng;
	 	}   
}
