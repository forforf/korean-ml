**(Rough draft and work in progress)**

### General Workflow

#### Pre-work (manual)

Given an audio file (currently just Korean audio) and a transcription,  we use Praat to generate a TextGrid 
that identifies characters, syllables and silences.
That TextGrid is then converted into a csv file that can be directly imported into Pandas.
This step is in the process of being automated (see `Textgrid Prediction`)


#### Feature Prep

code: `feature_prep.ipynb`

Find all kss-csvs, hold back ~20% for as the hold out set that will be used to score prediction algorithms.
The other 80% is used for the train/test.
The csv files are used to map hangul characters and symbols to the corresponding audio file. The characters/symbols
have associated timestamps.

The csv data is used to identify the timing of the features of interest from the underlying audio file as well as
the corresponding labels.

The features and labels are then saved to be used for training the models.

#### Model Training and Analysis

Given the names of desired models to analyze, the corresponding template specific for that model is looked up and
combined with the appropriate boiler-plate that is contained in the boiler-plate templates. A fully formed
executable notebook is then generated in the `models` directory. Running the notebooks will load the training data
saved during the `Feaure Prep` stage, train the model that corresponds to that notebook, and save the trained model.

Implementation Note: The actual model is abstracted in a wrapper. This wrapper includes any transformations
to be applied, trains the model, and allows for model prediction. The wrapper also provides model persistence and
versioning. 

#### *Why not just use SciKitLearn's Pipeline?*

The wrapper was built to support other types of model families (currently SciKitLearn and Keras). I ran into 
difficulties trying to build a Keras model into the ScikitLearn pipeline (one example was persistence, Keras models
are not pickle-able).

Having the wrapper allowed me to have identical analysis code regardless of the underlying model, and resulted in the 
ability to do apples-apples comparisons across models.


#### Testing/Scoring

Given a saved model (more precisely a model wrapper), the testing notebook will load the trained model and using the 
hold out set defined in the Feature Prep phase, it will plot and score the models. 

#### Textgrid Prediction

Given a model and feature set we predict the corresponding Praat textgrid.
If the generated textgrid is accurate it can be converted to a kss-csv file and incorporated into the train/test data.
Even if not completely accurate, it can still be a significant time saver as the generated textgrid is better than
generating a textgrid from scratch.

### KSS Event Detection using RMS Waveform

This analysis attempts to predict when there is speech (i.e. the speech event) vs when there is silence.

#### Feature Prep
The feature prep uses the syllable kss-csv file (the character one would work just as well, but the syllable 
is simpler). 

Silence is identified by `0` in the kss-csv file `syl` column and since for this analysis we only care about 
silence vs non-silence. 
The X is the RMS of the audio and the y is a boolean for every sample of the X data, that is 1 for speech and 0
for silence.

#### Training


## Challenges and Solutions

### General Approach
Time Series
Sliding Window Transformation
-> Pipeline
Worked well for SKLearn models, but then introduced Keras/CNN model
Pipeline wrapping Keras was annoying but possible
However, saving the model to disk was a problem with Keras as Keras models couldn't be persisted with joblib (pickle under the hood) 
-> Model Wrapper to abstract model save mechanism
This worked, but two (related) flaws found
- Required the transformer to be rebuilt when reloading the model
- Did not work with ensemble models (specifically, using a different transformation per model)
-> Improved Wrapping to incorporate 

### Custom Scoring Function 
The initial use case was to predict speech vs silence using from the audio waveform. 
- As this was time series data that was not IID (independent and identically distributed) many of the standard scoring 
  functions didn't perform well
- Also, for my use case, a false positive (predicting speech when it was actually silence) was worse than a false 
  negative (predicting silence when it was actually speech). As the predictions were used to split the waveform into
  fragments, and it was easier to join mistaken fragments than it was to find and split the audio when a period of 
  silence was missed.

### Scaling Model Comparisons
-> Templating Solution

### File Versioning for Saved Models
Wanted to be able to compare current model to older versions (version control such as git was not a good solution here 
as I wanted the older and newer versions to co-exist for direct comparison).



