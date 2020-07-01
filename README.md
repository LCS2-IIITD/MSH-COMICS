# Multi-modal Sarcasm Detection and HumorClassification in Code-mixed Conversations

This repository contains the dataset and code for our paper: Paper Link

We release the MaSaC dataset which is a multimodal Code-Mixed corpus(Hindi-English) for detection of sarcasm as well as humour. The dataset is compiled from popular hindi TV series Sarabhai vs Sarabhai. The dataset consists of code-mixed utterences which are accompanied by the speaker level information for those utterances. The utterances are annotated with respective sarcasm and humour labels. For every utterrence, we also include audio features with respect to the given utterrence, which are included to provide additional understanding for the utterance. The audio features are extracted on utterance level from the timestamp information anotated while collecting the dataset.   

Data Format


   #KEY                   #VALUE
  Speaker          Speaker for the utterance
  text             Utterance text to classify
  Audio_features   Extracted mfcc features from audio file corresponding to the current utterance
  Sarcasm          Binary label for sarcasm tag
  Humour           Binary label for humour tag



