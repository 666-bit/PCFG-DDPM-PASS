import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusion_utils import UNet, PasswordDataset, DiffusionModel, FCNet, ImprovedFCNet
import utils
import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    parser.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .pth file.')

    parser.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    parser.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the UNet (default: 128)')
    
    parser.add_argument('--num-timesteps', '-t',
                        type=int,
                        default=100,
                        dest='num_timesteps',
                        help='The number of diffusion timesteps (default: 100)')
    
    parser.add_argument('--beta-schedule', '-bs',
                        default='linear',
                        dest='beta_schedule',
                        help='The schedule for beta (default: linear)')
    
    parser.add_argument('--embedding-size', '-es',
                        type=int,
                        default=128,
                        dest='embedding_size',
                        help='The embedding size for the time (default: 128)')
    
    parser.add_argument('--learning-rate', '-lr',
                        type=float,
                        default=1e-4,
                        dest='learning_rate',
                        help='The learning rate for the optimizer (default: 1e-4)')
    
    parser.add_argument('--model', '-m',
                        type=str,
                        default='unet',
                        dest='model_name',
                        help='The model to use (default: unet)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error('"{}" folder doesn\'t exist'.format(args.input_dir))

    if not os.path.exists(os.path.join(args.input_dir, 'charmap.pickle')):
        parser.error('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    if not os.path.exists(os.path.join(args.input_dir, 'charmap_inv.pickle')):
        parser.error('charmap_inv.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))
    
    return args

def main():
    args = parse_args()

    # Dictionary
    with open(os.path.join(args.input_dir, 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f, encoding='latin1')

    # Reverse-Dictionary
    with open(os.path.join(args.input_dir, 'charmap_inv.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f, encoding='latin1') 

    # initialize model, optimizer and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == 'unet':
        model = UNet(in_channels=1, out_channels=1, time_dim=args.num_timesteps).to(device)
    elif args.model_name == 'fcnet':
        model = FCNet(seq_length=args.seq_length, vocab_size=len(charmap), embedding_size=args.embedding_size, hidden_size=args.layer_dim, timesteps=args.num_timesteps).to(device)
    elif args.model_name == 'res_fc':
        model = ImprovedFCNet(seq_length=args.seq_length, vocab_size=len(charmap), embedding_size=args.embedding_size, hidden_size=args.layer_dim, timesteps=args.num_timesteps).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    diffusion_model = DiffusionModel(model, num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule, device=device)

    # sample
    if not os.path.exists(os.path.join(args.input_dir, 'samples')):
        os.makedirs(os.path.join(args.input_dir, 'samples'))
    if not os.path.exists(os.path.join(args.input_dir, 'samples', args.model_name)):
        os.makedirs(os.path.join(args.input_dir, 'samples', args.model_name))
    with open(os.path.join(args.input_dir, 'samples', args.model_name, args.output), 'w') as f:
        for i in range(args.num_samples):
            samples = diffusion_model.sample(num_samples=1, seq_length=args.seq_length, vocab_size=len(charmap))
            decoded_samples = utils.decode_samples(samples, inv_charmap)
            f.write(decoded_samples[0] + '\n')
            if (i+1) % 100 == 0:
                print(f'Generated {i+1} samples, total {args.num_samples} samples.')

if __name__ == "__main__":
    main()
