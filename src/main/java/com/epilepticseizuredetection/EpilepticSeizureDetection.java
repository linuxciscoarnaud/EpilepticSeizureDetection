/**
 * 
 */
package com.epilepticseizuredetection;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.toIntExact;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class EpilepticSeizureDetection {

       protected static final Logger log = LoggerFactory.getLogger(EpilepticSeizureDetection.class);
    
       private Params params = new Params();
       private NetworkConfig networkConfig = new NetworkConfig();
       public static int numClasses = 0;
       private boolean save = true;
    
       public void execute(String[] args) throws Exception {
    	

        //Loading the data
        log.info("Loading data....");
		
	// Returns as label the base name of the parent directory of the data.
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    	// Gets the main path
        File mainPath = new File(System.getProperty("user.dir"), "/src/main/resources/checkedData1/");
        //File mainPath = new File("C:/TEMP/data/EpilepticData/trainingData1/");      
        // Split up a root directory in to files
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng()); 
        // Get the total number of images
        int numExamples = toIntExact(fileSplit.length());
        log.info("Total number of images: " + numExamples);
        // Gets the total number of classes
        // This only works if the root directory is clean, meaning it contains only label sub directories.
        numClasses = fileSplit.getRootDir().listFiles(File::isDirectory).length; 
        log.info("Number of classes: " + numClasses);
        // Randomizes the order of paths in an array and removes paths randomly to have the same number of paths for each label.
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numClasses, maxPathsPerLabel);       
        // Randomizes the order of paths of all the images in an array. (There is no attempt to have the same number of paths for each label, so there is no random paths removal).
        RandomPathFilter pathFilter = new RandomPathFilter(params.getRng(), null, numExamples); 
        // Gets the list of loadable locations exposed as an iterator.
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest); 
        // Gets the training data
        InputSplit trainData = inputSplit[0];
        log.info("Number of train data: " + trainData.length());
        // Gets the test data
        InputSplit testData = inputSplit[1];       
        log.info("Number of test data: " + testData.length());
        
        // Data transformation
        ImageTransform flipTransform = new FlipImageTransform(0);  // Flips around x-axis (flips horizontally).
        //ImageTransform warpTransform1 = new WarpImageTransform(0, 0, 3, 0, 3, 0, 0, 0); // Kind of deformation only in the horizontal direction
        //ImageTransform warpTransform2 = new WarpImageTransform(0, 0, 5, 0, 5, 0, 0, 0);
        boolean shuffle = true;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(flipTransform, 0.9));
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
        
       //Data normalization. Puts all the data in the same scale.
       DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
       // This will read a local file system and parse images of a given height and width.
       // All images are rescaled and converted to the given height, width, and number of channels.
       ImageRecordReader trainRecordReader = new ImageRecordReader(params.getHeight(), params.getWidth(), params.getChannels(), labelMaker);
       ImageRecordReader testRecordReader = new ImageRecordReader(params.getHeight(), params.getWidth(), params.getChannels(), labelMaker);
       // Train data will be initially transformed
       trainRecordReader.initialize(trainData, transform); 
       // Test data will not be initially transformed
       testRecordReader.initialize(testData);
        
       // Iterator which Traverses through the dataset and preparing the data for the network.
       DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, params.getMiniBatchSize(), 1, numClasses);
       scaler.fit(trainDataIter);
       trainDataIter.setPreProcessor(scaler);
       //log.info("SHAPE THE FEATURES: " + trainDataIter.next(2));
       DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, params.getMiniBatchSize(), 1, numClasses);
       scaler.fit(testDataIter);
       testDataIter.setPreProcessor(scaler);
       //log.info("SHAPE THE TEST: " + testDataIter.next(1));
        
       // Building model...
       log.info("Building model....");
       MultiLayerNetwork network;
       String modelFilename = "epilepticSeizureModel.zip";

       if (new File(modelFilename).exists()) {
                log.info("Loading existing model...");
                network = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
       } else {
        	network = networkConfig.getNetworkConfig();
        	network.addListeners(new ScoreIterationListener(10));
        	network.init();
        	log.info(network.summary(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels())));
        	
       // Enabling the UI...
       // Initialize the user interface backend.
       UIServer uiServer = UIServer.getInstance();
       // Configure where the information is to be stored. Here, we store in the memory.
       // It can also be saved in file using: StatsStorage statsStorage = new FileStatsStorage(File), if we intend to load and use it later.
       StatsStorage statsStorage = new InMemoryStatsStorage();
       // Attach the StatsStorage instance to the UI. This allows the contents of the StatsStorage to be visualized.
       uiServer.attach(statsStorage);
       // Add the StatsListener to collect information from the network, as it trains.
       network.setListeners(new StatsListener(statsStorage));
        	
       // Configuring early stopping for training
       EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        	.epochTerminationConditions(new MaxEpochsTerminationCondition(params.getEpochs()))
        	.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(params.getMaxTimeIterTerminationCondition(), TimeUnit.HOURS))
        	.scoreCalculator(new DataSetLossCalculator(testDataIter, true))
        	.evaluateEveryNEpochs(1)
        	.modelSaver(new LocalFileModelSaver(new File(System.getProperty("user.dir")).toString()))
        	.build();
        	
       EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, trainDataIter);
        	
       // Conducting early stopping training of the model...
       log.info("Training model....");
       EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
       log.info("Finishing training....");
			
       // Saving the best model...
       log.info("Saving the best model....");
       //Get the best model:
       network = result.getBestModel();
       if (save) {
	       ModelSerializer.writeModel(network, modelFilename, true);
       }
       log.info("Best model saved....");
	        
       // Getting the early stopping diagnostic ...
       log.info("Getting the early stopping diagnostic ...");
       log.info("Termination reason: " + result.getTerminationReason());
       log.info("Termination details: " + result.getTerminationDetails());
       log.info("Total epochs: " + result.getTotalEpochs());
       log.info("Best epoch number: " + result.getBestModelEpoch());
       log.info("Score at best epoch: " + result.getBestModelScore());
			System.out.println();
        } 
        
        // Evaluating model...
        log.info("Evaluating model...");
        Evaluation eval = network.evaluate(testDataIter);
        log.info(eval.stats(true));
    }
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		new EpilepticSeizureDetection().execute(args);
	}
}
