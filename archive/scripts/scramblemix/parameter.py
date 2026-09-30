import argparse

def get_parameters():

    parser = argparse.ArgumentParser()

    # For Help
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--milestones', default='75,125', type=str)

    # For Networks
    parser.add_argument("--depth", type=int, default=110)
    parser.add_argument("--alpha", type=int, default=270)
    parser.add_argument("--label", type=int, default=10)

    # For Training
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument('--e', '-e', default=150, type=int, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_of_keys", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_of_TTA", type=int, default=1)
    parser.add_argument("--js_divergence_regularization", type=bool, default=False)
    # file name
    parser.add_argument("--tensorboard_name", type=str, default="tensorboard")
    parser.add_argument("--training_model_name", type=str, default="model.t7")
    parser.add_argument("--json_file_name", type=str, default="results.json")
    parser.add_argument("--save_directory_name", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model_name", type=str, default="resnet18")
    

    return parser.parse_args()

