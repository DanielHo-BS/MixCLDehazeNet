# Set the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MixDehazeNet

# Set common variables
DATAPATH="../datasets/data/"
DATASET="RESIDE-6K"
EXP="reside6k"
GPUS="0,1"

# Create the text features for the dataset
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS

# Train the model
# Dataset directory: ../dataset/data/
#python train.py --model MixDehazeNet-b --save_dir ./saved_models/indoor/ --dataset RESIDE-IN --exp indoor --gpu 2
#python train.py --model MixDehazeNet-b --save_dir ./saved_models/outdoor/ --dataset RESIDE-OUT --exp outdoor --gpu 3
python train.py --model MixDehazeNet-s --save_dir ./saved_models/reside6k/ --data_dir $DATAPATH --dataset $DATASET --exp $EXP --gpu $GPUS
