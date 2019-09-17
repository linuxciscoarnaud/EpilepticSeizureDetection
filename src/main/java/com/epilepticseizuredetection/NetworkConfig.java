/**
 * 
 */
package com.epilepticseizuredetection;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Arnaud
 *
 */

public class NetworkConfig {

	Params params = new Params();
	
	public MultiLayerNetwork getNetworkConfig() {
		
		// LeNet config
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(params.getSeed())
				.optimizationAlgo(params.getOptimizationAlgorithm())
				.activation(params.getActivation())
                .weightInit(params.getWeightInit())
                .updater(params.getUpdater())
                .cacheMode(params.getCacheMode())
                .trainingWorkspaceMode(params.getWorkspaceMode())
                .inferenceWorkspaceMode(params.getWorkspaceMode())
                .cudnnAlgoMode(params.getCudnnAlgoMode())
                .convolutionMode(ConvolutionMode.Same)
				.list()
				
				// block 1
				.layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1})
						.name("cnn1")
						.activation(Activation.RELU)
                        .nIn(params.getChannels())
                        .nOut(20)                        
                        .build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2}, new int[] {2, 2})
						.name("maxpool1")
						.build())
				
				// block 2
				.layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1})
						.name("cnn2")
						.activation(Activation.RELU)
                        .nOut(50)                        
                        .build())
				.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2}, new int[] {2, 2})
						.name("maxpool2")
						.build())
				
				// fully connected
				.layer(4, new DenseLayer.Builder()
						.name("ffn1")
						.activation(Activation.RELU)
						.nOut(500)
						.build())
				
				// output
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.name("output")
                        .nOut(EpilepticSeizureDetection.numClasses)
                        .activation(Activation.SOFTMAX) // radial basis function required
                        .build())
				
				.setInputType(InputType.convolutionalFlat(params.getWidth(), params.getHeight(), params.getChannels()))
				.build();
				
		return new MultiLayerNetwork(conf);
	}
}
