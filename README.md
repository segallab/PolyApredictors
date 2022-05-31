# PolyApredictors
## Setup
### Clone repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/segallab/PolyApredictors.git
```

### Requirements
To run `EndogenousCleavageSitePrediction.py` please make sure you can satisfy the following dependencies: 

- python 2.7.8
- keras 2.0.6
- cPickle 1.71
- joblib 0.11
- pandas 0.23.4
- numpy 1.14.2
- seaborn 0.7.1

YAMLs are provided for creating a Conda environment with compatible dependencies installed:

```bash
conda create -n polyApredictors -f envs/polyApredictors.min.yaml
conda activate polyApredictors
```

## Running
The `EndogenousCleavageSitePrediction.py` script loads **Supplemental_Table_8.tab** and performs cleavage site prediction using each of the three models: **cnn**, **xgb**, **kmer**. For each model, the input is processed according to model requirements, the model is loaded and predictions are run. The predicted values are compared to the ones in **Supplemental_Table_8.tab**.

To predict the cleavage position for a new sequence, please run:
```bash
python PredictForNewSequence.py '<model_name>' '<dna_sequece>'
```

where:
* `<model_name>` should be replaced by `cnn`, `xgb` or `kmer`. The **cnn** model has previously shown the best performance.
* `<dna_sequence>` should be replaced by the DNA sequence. The reporter cleavage site is relative to sequence start. The indexes are zero based.
