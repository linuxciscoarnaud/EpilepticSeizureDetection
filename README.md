# Epileptic Seizure Detection using a CNN and implemented using the Deeplearning4j library.

I show here how we can use Convolutional Neural Network, implemented with Deeplearning4j library to detect Epileptic Seizure from images generated from EEG (electroencephalogram) signals. This implementation follows the original work presented in the paper [Epileptic Seizure Detection Using a Convolutional Neural Network](http://oatao.univ-toulouse.fr/24138/).

# Network Architecture
LeNet architecture was used. Details about this architecture can be found [here](https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf).

# The Dataset

Data used to validate this work comes from the database collected at the [Children’s Hospital Boston](https://archive.physionet.org/pn6/chbmit/) which consists of bipolar EEG recordings from pediatric subjects with intractable seizures. These data are subsequently pre-processed as follow:
- From the continuous EEG signals (which include all the 23 electrodes used to make the recording), a fixed temporal portion of 2 seconds is extracted. For records containing a seizure, this temporal portion of 2 seconds is extacted in such a way that the time during which the seizure was observed is included.
- Intensity images a then generated from each of the fixed temporal portion of 2 seconds previously extracted.

![sample data](https://user-images.githubusercontent.com/1300982/65240346-f03f9280-dad8-11e9-81d6-850f34be12f5.png)

The intensity images are used as input to the CNN.
