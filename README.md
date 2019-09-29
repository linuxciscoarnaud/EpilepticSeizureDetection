# Epileptic Seizure Detection using a CNN and implemented using the Deeplearning4j library.

I show here how we can use Convolutional Neural Network, implemented with Deeplearning4j library to detect Epileptic Seizure from images generated from EEG (electroencephalogram) signals. This implementation follows the original work presented in the paper [Epileptic Seizure Detection Using a Convolutional Neural Network](http://oatao.univ-toulouse.fr/24138/).

# Epileptic seizure
An epileptic seizure is defined as a disruption of the electrical activity of the humain brain. This activity is normally picked up by an electroencephalograph (EEG) when the brain is performing a cognitive task. When an epileptic seizure occurs, the normal pattern of the brain activity that is seen by the EEG reading changes and different brain activity can be seen. The localization of this change (in other words the localization of normal or abnormal brain waves for epileptic seizure detection) in a sequential bipolar montage is accomplished by identifying "phase reversal". Phase reversal is a concept used in clinical neurophysiology and is often  reffered to as the oppisite simultaneous deflection of pens in the channels that contain a common electrode. We will be relying on this concept to extract learnable features from images that will be used as inputs to our CNN.

# Network Architecture
LeNet architecture was used. Details about this architecture can be found [here](https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf).

# The training data
The data used to validate this work comes from the database collected at the [Children’s Hospital Boston](https://archive.physionet.org/pn6/chbmit/) which consists of bipolar EEG recordings from pediatric subjects with intractable seizures. These data are subsequently pre-processed as follow: 
- From the continuous EEG signals (which include all the 23 electrodes used to make the recording), fixed temporal portions/windows of 2     seconds are extracted, as shown on the following picture:
  ![2SECS](https://user-images.githubusercontent.com/1300982/65584531-f02c1080-df78-11e9-9474-39a44bc1c81a.png)
  
- Intensity images are generated from the extracted portions of 2 seconds. For data containing epileptic seizures, the phase reversal     that i talked about earlier and that can be easilly observed (for some cases) on the EEG signals are replicated on the intensity images, as shown on the following figure.  
 ![show](https://user-images.githubusercontent.com/1300982/65589248-ac3d0980-df80-11e9-9f2b-e162958566d1.png)
 You can see that channels "F8-T8, T8-P8_02" and "FT10-T8, T8-P8" that contain common electrodes (T8 for the two cases) repeatedly have     opposite and simultaneous deflections of pens. These phase reversal, which have been highlighted on the generated intensity image can therefore be considered as features to be learned by a CNN. The goal being to be able to come up with a model, which can then be used to detect epileptic seizures on other unseen generated images.

# Get the code to work


