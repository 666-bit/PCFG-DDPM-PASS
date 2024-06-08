import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusion_utils import UNet, PasswordDataset, DiffusionModel, FCNet, ImprovedFCNet
import utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')
    
    parser.add_argument('--checkpoint', '-c',
                        type=str,
                        default=None,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .pth file.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--num-epochs', '-e',
                        type=int,
                        default=10,
                        dest='num_epochs',
                        help='The number of training epochs (default: 10)')

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
                        default='fcnet',
                        dest='model_name',
                        help='The model to use (default: unet)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.training_data,
        max_length=args.seq_length
    )

    os.makedirs(args.output_dir, exist_ok=True)
    utils.save_vocab(charmap, inv_charmap, args.output_dir)

    print(f"Number of unique characters in dataset: {len(charmap)}")

    dataset = PasswordDataset(lines, charmap)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize model, optimizer and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == 'unet':
        model = UNet(in_channels=1, out_channels=1, time_dim=args.num_timesteps).to(device)
    elif args.model_name == 'fcnet':
        model = FCNet(seq_length=args.seq_length, vocab_size=len(charmap), embedding_size=args.embedding_size, hidden_size=args.layer_dim, timesteps=args.num_timesteps).to(device)
    elif args.model_name == 'res_fc':
        model = ImprovedFCNet(seq_length=args.seq_length, vocab_size=len(charmap), embedding_size=args.embedding_size, hidden_size=args.layer_dim, timesteps=args.num_timesteps).to(device)
    
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    diffusion_model = DiffusionModel(model, num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule, device=device)

    # train
    diffusion_model.train(dataloader, optimizer, criterion, args.num_epochs)

    # sample
    samples = diffusion_model.sample(num_samples=10, seq_length=args.seq_length, vocab_size=len(charmap))
    decoded_samples = utils.decode_samples(samples, inv_charmap)
    for sample in decoded_samples:
        print(sample)

    # save model
    if not os.path.exists(os.path.join(args.output_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', f'ddpm_{args.model_name}.pth'))

if __name__ == "__main__":
    main()
