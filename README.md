[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hb20007/cpp-programs/blob/master/LICENSE.md)

# ![](resources/favicon.png) Genre-It  

*The mobile app that recognizes music genres using Deep Learning*

## Description

### How it works

#### Why CNN and LSTM? 
* CNN makes sense since spectograms look like an image, each with their own distinct patterns. 
* RNNs excel in understand sequential data by making the hidden state at time t dependent on hidden state at time t-1. 
* The spectograms have a time component and RNNs can do a much better job of identifying the short term and longer term temporal features in the song.

#### Parallel CNN-RNN Model
* The model passes the input spectogram through both CNN and RNN layers in parallel, concatenating their output and then sending this through a dense layer with softmax activation to perform classification.
* The **convolutional** block of the model consists of 2D convolution layer followed by a 2D Max pooling layer. There are 5 blocks of Convolution Max pooling layers. The final output is flattened and is a tensor of shape None , 256.
* The **recurrent block** starts with 2D max pooling layer of pool size 4,2 to reduce the size of the spectogram before LSTM operation. This feature reduction was done primarily to speed up processing. The reduced image is sent to a bidirectional GRU with 64 units. The output from this layer is a
tensor of shape None, 128.
* The **outputs** from the convolutional and recurrent blocks are then concatenated resulting in a tensor of shape, None, 384. Finally we have a dense layer with softmax activation.

#### Model accuracy
* The accuracy of the model is around 51%
* The reason for this is the small sample size of spectograms, which is a very small sample for building a deep learning neural network.
* The FMA data set is challenging and has few classes which are easy to confuse among. The top leader board score on FMA-Genre Recongnition has a test F1 score of around 0.63
* The accuracy is much better than guessing in random, which would be around 0.125 and could be improved if additional datasets and hardware was available.

### Genres

1\. Electronic  
2\. Experimental  
3\. Folk  
4\. Hip-Hop  
5\. Instrumental  
6\. International  
7\. Pop  
8\. Rock  

## Getting Started

### Prerequisites

* [Python 3.4+](https://www.python.org/)

### Installing

* `pip install -r requirements.txt`

## Built with

* [Python](https://www.python.org/)
* [Keras](https://keras.io/) — Deep Learning — Neural Networks
* [Librosa](https://librosa.github.io/librosa/) — Sound processing

## Tested on
* [Linux Debian](https://www.debian.org/)
* Running on Windows machines could yield issues, due to missing codecs*

## Datasets used for training

 * [Free Music Archive](https://github.com/mdeff/fma)

## Authors

* **Hanna Sababa** — Full Stack Developer — [hb20007](https://github.com/hb20007)
* **Chris Peppos** — Full Stack Developer, Music Expert — [ChrisPeppos](https://github.com/ChrisPeppos)
* **Karlen Avogian** — Deep Learning, Cyber Security — [kosnet2](https://github.com/kosnet2)


## Acknowledgments

* Kudos to [Hack{Cyprus}](http://comeback.hackcyprus.com/) for organizing the event.
* Kudos to anyone whose code and ideas were used in our project.
* Kudos to all the open source community for allowing us achieve the impossible.