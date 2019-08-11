# PolyApredictors
First, clone the repository to your local machine.

To run EndogenousCleavageSitePrediction.py please make sure you can satisfy the following dependencies: python 2.7.8, keras version 2.0.6, cPickle version 1.71, joblib version 0.11, pandas version 0.23.4, numpy version 1.14.2, seaborn version 0.7.1. The script loads Supplemental_Table_8.tab and performs cleavage site prediction using each of the three models: cnn, xgb, kmer. For each model, the input is processed according to model requirements, the model is loaded and predictions are run. The predicted values are compared to the ones in Supplemental_Table_8.tab.

To predict the cleavage position for a new sequence, please run:
```
python PredictForNewSequence.py '<model_name>' '<dna_sequece>'
```
Where:
* <model_name> should be replaced by cnn, xgb or kmer. The cnn model has previously shown the best performance.
* <dna_sequence> should be replaced by the DNA sequence. The reporter cleavage site is relative to sequence start. The indexes are zero based.
