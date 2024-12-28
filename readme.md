# EGonc:Energy-based Open-Set Node Classification with substitute Unknowns

## Requirement

- python 3.8.11
- pytorch 1.7.1
- torch_geometric 1.7.2
- ogb 1.3.4
---
## Pretrain
Specify the dataset, along with the relevant parameters and the storage path for the pre-training results, then run the following code for pre-training.

`Python Pretrain_Egonc_gcn_model2_bn_inductive.py`
## Finetune
After the pre-training is complete and the model's pre-training parameters are obtained, run the following code for fine-tuning to get the test results.

`Python Finetune_Egonc_gcn_model2_bn_inductive`
