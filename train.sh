# Set the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MixDehazeNet

# Train the model
# Dataset directory: ../dataset/data/
python train.py --model MixDehazeNet-b --save_dir ./saved_models/indoor/ --dataset RESIDE-IN --exp indoor --gpu 2

python train.py --model MixDehazeNet-b --save_dir ./saved_models/outdoor/ --dataset RESIDE-OUT --exp outdoor --gpu 3

python train.py --model MixDehazeNet-s --save_dir ./saved_models/reside6k/ --dataset RESIDE-6K --exp reside6k --gpu 2,3

