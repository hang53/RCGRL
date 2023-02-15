# RCGRL
The codes of Robust Causal Graph Representation Learning against Confounding Effects

Please use the python scripts in /RCGRL/spmotif_gen/ to generate the raw data first, then run /RCGRL/train/sp-mtf_rcgrl.py to repeat our experiments. Specificially:

Run python3 /RCGRL/spmotif_gen/spmotif_test_dataset_gen.py to generate test.npy 
Run python3 /RCGRL/spmotif_gen/spmotif_train_dataset_gen.py to generate the train.npy 
Run python3 /RCGRL/spmotif_gen/spmotif_validate_dataset_gen.py to generate the val.npy 

then put test.npy, train.npy, val.npy into /RCGRL/data/SPMotif-0.9/raw/, and run:
python3 /RCGRL/train/sp-mtf_rcgrl.py


