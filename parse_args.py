from argparse import ArgumentParser

def _clear_args(parsed_args):
    parsed_args.experiment_args = eval(parsed_args.experiment_args)
    parsed_args.dataset_args = eval(parsed_args.dataset_args)
    return parsed_args

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Seed used for deterministic behavior')
    parser.add_argument('--test_only', action='store_true', help='Whether to skip training')
    parser.add_argument('--cpu', action='store_true', help='Whether to force the usage of CPU')

    parser.add_argument('--experiment', type=str, default='baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--experiment_args', type=str, default='{}')
    parser.add_argument('--dataset_args', type=str, default='{}')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--grad_accum_steps', type=int, default=1)

    parser.add_argument('--topK', action='store_true', help='Wheter to adapt activation map to output topKs')
    parser.add_argument('--tk_treshold', type=float, default=1, help='If topK is enabled, this controls K (how many elements will be retained in the activation map)')
    parser.add_argument('--no_binarize', action='store_true', help='Wheter to keep activation mask as it is, without binarizing it')
    parser.add_argument('--mask_ratio', type=float, default=1, help='If the experiment is random_maps, this controls the ratio of 1s in the random mask')

    return _clear_args(parser.parse_args())