import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=5)
    parser.add_argument('--alp', type=float, default=0.9)
    # parser.add_argument('--input_c', type=int, default=7)
    # parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--anomaly', type=int, default=30)
    # parser.add_argument('--dataset_name', type=str, default='user')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='latency')
    parser.add_argument('--mode', type=str, default='train', choices=['pretrain', 'train', 'test', 'rca', 'pre_rca'])
    parser.add_argument('--data_path', type=str, default='./dataset/processed/SMD/')
    parser.add_argument('--scheme', type=str, default='ascending', choices=['ascending', 'descending'])
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)#冗余
    parser.add_argument('--adj', type=float, default=1.00)
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--alpha", type=float,default=0.7, choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], help="DRscore weight")
    parser.add_argument("--beta", type=float,default=1, choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], help="Predict weight")

    parser.add_argument("--seed", type=int,default=1, choices=[1, 2, 3, 4, 5], help="random seed")
    parser.add_argument("--horizon", type=int,default=1, choices=[1, 2, 3, 4, 5], help="predict horizon")


    return parser