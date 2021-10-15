# Korean ML Workspaces

A project to develop automation to accurately predict Korean language (hangul) from [Korean Single Speaker
Speech Dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset.

This is a hobby project of mine so the true goal of the project is basically to educate and entertain myself.

## Overview and Current Status

The existing KSS dataset is not ideal for predictions. The audio and transcriptions are for full sentences/phrases
and is not sufficient for character prediction. What is missing is a mapping of the audio data to the occurrence
of specific characters and syllables. Fortunately there is a tool called [Praat](https://praat.org) that can analyze audio
files and segment them into sections and assign labels to those segmented sections and output that data into a specific
file format (a "TextGrid" file). 

### Phase 1

Generating TextGrid files through Praat is a manual process. As a first stage in automating this process, a seed group 
of 6 audio files was manually segmented and then used  as the training/test set to identify speech vs non-speech 
segments of the audio file and to programmatically generate base TextGrid files. This step would reduce the amount of
manual steps and increase the turn around time generating the fully segmented TextGrid files that can be used for
training data.

#### Status 2015-10-15
Completed development on the initial workflow feedback loop that uses predictive tools to assist in generating
the labeled training data. The workflow loop:

1. Given a set of Audio Files and a corresponding segmented TextGrid files
2. Use the TextGrid files Generate Character and Syllable level CSV files
3. Import the CSV files into Pandas Dataframes
4. Use the Audio Files and Character Dataframe to generate features (RMS audio waveform) and labels (speech / no speech)
5. Remove a fraction (~20%) of the features/labels to use as a hold out set for analysis and scoring.
6. Use the remaining features and labels as the training/test for multiple predictive models
7. Train the models of interest
8. Analyze and Score the models on the hold out set
9. Use a subset of the best performing models to predict on a new audio file
10. [Manual] Inspect the predictions and choose the best predictor, tweak the scoring algorithm as needed.
11. Use the best predictor's predictions to generate a new TextGrid file
12. [Manual] Update the generated TextGrid file with character and syllable segmentations
13. Add the new file to the set of Audio Files and TextGrid files used for train/test
14. Goto Step 1

The predictions are okay, but not yet high quality enough to move to Phase 2. This is not unexpected as the training
size is still relatively small (6-7 audio files). Hopefully once ~20 files have been added the predictions will have 
improved enough (fingers crossed).

### Phase 2
Once the speech/no-speech analysis is working well enough (i.e., the silence predictions are accurate enough) 
convert to predicting characters or syllables (depending on which works better). The general workflow would remain the
same.

## [KSS Data Exploration](kss-exploration)

### [KSS Corpus Analysis](kss-exploration/kss-corpus.ipynb)

Some statistics and analysis of the KSS transcription data.

## [KSS Event Detection from Audio RMS](kss-event-detection-rms)


Workspace for code and notebooks for predicting speech/no speech from the KSS Audio files and transcription data.

[Workspace README](kss-event-detection-rms/_README.md)



