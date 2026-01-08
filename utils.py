import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_training_args():
    parser = argparse.ArgumentParser(description='SEHA')

    parser.add_argument("--dataset", type=str, default="xmedia", help="Dataset to use (wiki, xmedia, INRIA-Websearch, xmedianet)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lamda", type=float, default=0.005)
    parser.add_argument("--MAX_EPOCH", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bit", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--noisy_ratio", type=float, default=0.4) # (0.2, 0.4, 0.6, 0.8)
    parser.add_argument("--noise_mode", type=str, default='sym')
    parser.add_argument("--tp", type=int, default=5)
    parser.add_argument("--q", type=float, default=0.01)
    parser.add_argument("--loss_type", type=str, default='CE')
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument('--linear', type=str2bool, default=True)
    parser.add_argument("--GPU", type=int, default=0)

    args = parser.parse_args()

    return args