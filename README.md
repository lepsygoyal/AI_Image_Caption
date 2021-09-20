# AI Image Captioning (Deep Learning)
Implementation of 'merge' architecture for generating image caption from paper "What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?" using Keras. Dataset used is Flickr8k available on Kaggle.
<hr>

## Some sample captions that are generated
<img src="https://github.com/lepsygoyal/AI_Image_Caption/blob/main/Images/Img_1.png" width="950" height="500" />

<img src="https://github.com/lepsygoyal/AI_Image_Caption/blob/main/Images/Img_3.png" width="950" height="500" />

<img src="https://github.com/lepsygoyal/AI_Image_Caption/blob/main/Images/Img_2.png" width="950" height="500" />

<hr>

## Getting Started
Here I will explain about the 'merge' architecture that is used for generation image caption. 
<hr>

## Prerequisites
You will need Python 3.X.X with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt
<hr>

## Get the Dataset
I have used flickr8k dataset that is available on kaggle.
<hr>

## Generating Image Caption
In neural image captioning systems, a recurrent neural network (RNN) is typically viewed as the primary ‘generation’ component. This view suggests that the image features should be ‘injected’ into the RNN. This is in fact the dominant view in the literature. Alternatively, the RNN can instead be viewed as only encoding the previously generated words. This view suggests that the RNN should only be used to encode linguistic features and that only the final representation should be ‘merged’ with the image features at a later stage. Here I will explain 'merge' architecture.

This architecture keeps the encoding of linguistic and perceptual features separate, merging them in a later multimodal layer, at which point predictions are made. In this type of model, the RNN is functioning primarily as an encoder of sequences of word embeddings, with the visual features merged with the linguistic features in a later, multimodal layer. This multimodal layer is the one that drives the generation process since the RNN never sees the image and hence would not be able to direct the generation process.

In this model, RNN is only used as language model. RNN is feeded the word embeddings of partial caption starting from special token 'seq_start', the RNN then generate encoded representation of partial sequence. While CNN is feeded the image, which generate a image representation.
These two representation i.e. language feature and image feature are appended together and feeded into another Feed Forward neural network. This FNN will output a vector of size equal to size of vocabulary. Index of highest value in that vector represents the next word of caption which is combined with the partial caption and again process continues untill we get the 'seq_end' token from FNN.

According to the paper, ["What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?"](https://arxiv.org/pdf/1708.02043.pdf), in general, for image captioning task it is better to have a RNN that only performs word encoding. In short for generation task, involving sequence it is a better idea to have a separate network to encode each input data rather than to give everything to the RNN.

<hr>

## Architecuture used in Notebook:

<img src="https://github.com/lepsygoyal/AI_Image_Caption/blob/main/Architecture.png" width="750" height="600" />

<hr>

## BLEU Score
BLEU, or the Bilingual Evaluation Understudy, is a score for comparing a candidate translation of text to one or more reference translations. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.

The primary programming task for a BLEU implementor is to compare n-grams of the candidate with the n-grams of the reference translation and count the number of matches. These matches are position-independent. The more the matches, the better the candidate translation is. Unfortunately, MT systems can overgenerate “reasonable” words, resulting in improbable, but high-precision, translations. Intuitively the problem is clear: a reference word should be considered exhausted after a matching candidate word is identified. We formalize this intuition as the modified unigram precision.

We first compute the n-gram matches sentence by sentence. Next, we add the clipped n-gram counts for all the candidate sentences and divide by the number of candidate n-grams in the test corpus to compute a modified precision score, pn, for the entire test corpus.

The BLEU metric ranges from 0 to 1. Few translations will attain a score of 1 unless they are identical to a reference translation. For this reason, even a human translator will not necessarily score 1.

NLTK provides the sentence_bleu() function for evaluating a candidate sentence against one or more reference sentences. NLTK also provides a function called corpus_bleu() for calculating the BLEU score for multiple sentences such as a paragraph or a document.

<hr>

## Authors
Lepsy Goyal

