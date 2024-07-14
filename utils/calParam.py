import argparse
import torch
from thop import profile
from models import *


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('==> Using device:', device)
    
    # Load the model
    network = eval((args.model).replace('-', '_'))()
    network.to(device)
    
    # Init the inupt tensor
    x = torch.randn(1, 3, 256, 256).to(device)
    x1 = torch.randn(1, 1, 512).to(device)

    # Calculate the number of parameters and MACs
    macs, params = profile(network, inputs=(x, x1))  # ,verbose=False
    print("MACs", macs)
    print("p", params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='model name')
    args = parser.parse_args()

    main(args)