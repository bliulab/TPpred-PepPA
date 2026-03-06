# TPpred-PepPA
The implementation of the paper ***Peptide Language Pragmatic Analysis and Two-stage Hierarchical Learning framework for Therapeutic Peptide Prediction PLPTP: A Motif-based Interpretable Deep Learning Framework Based on Protein Language Models for Peptide Toxicity Prediction***

## Requirements
The majoy dependencies used in this project are as following:

```
python  3.7
numpy 1.21.6
tqdm  4.64.1
pyyaml  6.0
scikit-learn  1.0.2
torch  1.11.0+cu113
tensorflow  1.14.0
tensorboardX  2.5.1
transformers  4.25.1
```
More detailed python libraries used in this project are referred to `requirements.txt`. 

# Usage
+ Generate the pssms by blast against NR database(https://ftp.ncbi.nlm.nih.gov/blast/db/). The features of the benchmark data are avilable at (http://bliulab.net/TPpred-PepPA/data/).
+ Create the `features`,  `logs`, `results` directory in current path. 
+ copy the `pssm` into features. It should be `/features/pssm/xxx.pssm`. 
+ train and test the model:
Train the model:
```shell
./train.sh
```

