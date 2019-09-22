# Epileptic Seizure Detection using a CNN and implemented using the Deeplearning4j library.

I show here how we can use Convolutional Neural Network, implemented with Deeplearning4j library to detect Epileptic Seizure from images generated from EEG (electroencephalogram) signals. This implementation follows the original work presented in the paper [Epileptic Seizure Detection Using a Convolutional Neural Network](http://oatao.univ-toulouse.fr/24138/).

# Epileptic seizure
An epileptic seizure is defined as a disruption of the electrical activity of the humain brain. This activity is picked up by electroencephalograph (EEG) when the brain is performing a cognitive task. When an epileptic seizure occurs, the normal pattern of the brain activity that is seen by the EEG reading changes and different brain activity can be seen.

# Network Architecture
LeNet architecture was used. Details about this architecture can be found [here](https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf).

# The training data
Data used to validate this work comes from the database collected at the [Children’s Hospital Boston](https://archive.physionet.org/pn6/chbmit/) which consists of bipolar EEG recordings from pediatric subjects with intractable seizures. These data are subsequently pre-processed as follow: From the continuous EEG signals (which include all the 23 electrodes used to make the recording), Intensity images are generated. These images contain features corresponding to the disruptions i talked about earlier. Although for some cases these features are not so visually perceptible (see figure below), they can be learned by a CNN.

![Untitled](https://user-images.githubusercontent.com/1300982/65384096-0dad7000-dd16-11e9-8027-2f0e9ced596c.png)


