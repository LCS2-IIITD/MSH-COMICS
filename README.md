# Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations

This repository contains the dataset and code for our paper: Paper Link

We release the MaSaC dataset which is a multimodal Code-Mixed corpus(Hindi-English) for detection of sarcasm as well as humour. The dataset is compiled from popular hindi TV series Sarabhai vs Sarabhai. The dataset consists of code-mixed utterences which are accompanied by the speaker level information for those utterances. The utterances are annotated with respective sarcasm and humour labels. For every utterrence, we also include audio features with respect to the given utterrence, which are included to provide additional understanding for the utterance. The audio features are extracted on utterance level from the timestamp information anotated while collecting the dataset.   

## Data Format


  |KEY           |          VALUE                                                                  |
  |--------------|---------------------------------------------------------------------------------|
  |Speaker       |   Speaker for the utterance                                                     |
  |text          |   Utterance text to classify                                                    |
  |Audio_features|   Extracted mfcc features from audio file corresponding to the current utterance|
  |Sarcasm       |   Binary label for sarcasm tag                                                  |
  |Humour        |   Binary label for humour tag                                                   |
  
  
## Raw Audios
  
The raw audio files for each utterance is provided in the Google drive folder. For each utterence, the name structure for the audio files can be found in the "Audio_Filename.txt" file. In addition, the episode wise dialogue information is also provided in the file "Episodewise_dialoguelabels.txt" to identify the set of dialogues to which a particular utterance belongs. The audio features are also included seperately in a pickle file which are pre loaded by the model.

## Running the Code 

Download the pre-trained [Fasttext](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md) multilingual word embeddings anywhere in the directory.

Extract the embedding matrix file to get the pickled version.

-Check for the configuration from the config.py file as per convinience.



