# LGCMT

## Dataset Preparation
We provide the pre-processed datasets used in this study (ETH, UCY, SDD, etc.) via Zenodo.
Download Link: https://zenodo.org/records/15691159

## Training
You can train the model using train.py. You need to specify the dataset name and the path to the hyper-parameter configuration file.
python train.py --dataset_name <name> --gpu <gpu_id> --hp_config <path_to_config>

## Note
All code required to reproduce the main results is already available; additional tools/weights will be released upon publication.
