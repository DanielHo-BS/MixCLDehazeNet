# Set the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MixDehazeNet

# Set common variables
DATAPATH="../datasets/data/"
DATASET="RESIDE-6K"
GPUS="0"
MODEL="'ViT-B/32'"

# Create the text features for the dataset
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model RN50
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model RN101
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model RN50x4
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model RN50x16
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model RN50x64
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model ViT-B/32
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model ViT-B/16
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model ViT-L/14
python clip_text.py --data_dir $DATAPATH --dataset $DATASET --gpu $GPUS  --model ViT-L/14@336px