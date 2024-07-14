# Set the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MixDehazeNet

# Set common variables
SAVEPATH="./saved_models/ParametricAtten2_5/"
DATAPATH="../datasets/data/"
DATASET="RESIDE-6K"
EXP="reside6k"
GPUS="0"

# Create the text features for the dataset
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS

# Train the model
# Dataset directory: ../dataset/data/
python train.py --model MixDehazeNet-s --save_dir $SAVEPATH --data_dir $DATAPATH --dataset $DATASET --exp $EXP --gpu $GPUS

# Evaluate the model
python test.py --model MixDehazeNet-s --save_dir $SAVEPATH --data_dir $DATAPATH --dataset $DATASET --exp $EXP --gpu $GPUS