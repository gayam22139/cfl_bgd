import argparse
import utils.datasets
import optimizers_lib
import models

def get_args():
    
    ###########################################################################
    # Script's arguments
    ###########################################################################

    archs_names = sorted(name for name in models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(models.__dict__[name]))
    optimizers_names = sorted(name for name in optimizers_lib.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(optimizers_lib.__dict__[name]))
    datasets_names = sorted(name for name in utils.datasets.__dict__
                            if name.islower() and name.startswith("ds_")
                            and callable(utils.datasets.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='Train and record statistics of a Neural Network with BGD')

    parser.add_argument('--dataset', default="ds_mnist", type=str, choices=datasets_names,
                        help='The name of the dataset to train. [Default: ds_mnist]')
    parser.add_argument('--nn_arch', type=str, required=True, choices=archs_names,
                        help='Neural network architecture')
    parser.add_argument('--logname', type=str, required=False,
                        help='Prefix of logfile name')
    parser.add_argument('--results_dir', type=str, default="TMP",
                        help='Results dir name')
    parser.add_argument('--seed', type=int,
                        help='Seed for randomization.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Num of workers for data loader')
    parser.add_argument('--num_epochs', default=400, type=int,
                        help='Maximum number of training epochs. [Default: 400]')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size [Default: 128]')
    parser.add_argument('--pruning_percents', default=[], type=int, nargs='*',
                        help='A list of percents to check pruning with [Default: []]')

    # BGD
    parser.add_argument('--train_mc_iters', default=10, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--std_init', default=5e-2, type=float,
                        help='STD init value (default 5e-2)')
    parser.add_argument('--mean_eta', default=1, type=float,
                        help='Eta for mean step (default 1)')

    parser.add_argument('--permanent_prune_on_epoch', default=-1, type=int,
                        help='Permanent prune on epoch')
    parser.add_argument('--permanent_prune_on_epoch_percent', default=90, type=float,
                        help='Permanent prune percent of weights')

    # SGD
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)


    #This tells us to check accuracy after every x epochs(x is the value of test_freq)
    parser.add_argument('--test_freq', default=1, type=int,
                        help='Run test set every test_freq epochs [default: 1]')
    parser.add_argument('--contpermuted_beta', default=3, type=int,
                        help='Beta value for continuous permuted mnist. [default: 3]')

    parser.add_argument('--optimizer', type=str, default="bgd", choices=optimizers_names,
                        help='Optimizer.')
    parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                        help='Optimizer parameters')


    parser.add_argument('--inference_mc', default=False, action='store_true',
                        help='Use MonteCarlo samples as inference method.')
    parser.add_argument('--inference_map', default=False, action='store_true',
                        help='Use MAP as inference method.')
    parser.add_argument('--inference_committee', default=False, action='store_true',
                        help='Use committee as inference method.')
    parser.add_argument('--inference_aggsoftmax', default=False, action='store_true',
                        help='Use aggsoftmax as inference method.')
    parser.add_argument('--inference_initstd', default=False, action='store_true',
                        help='Use initstd as inference method.')


    parser.add_argument('--committee_size', default=0, type=int,
                        help='Size of committee (when using committee inference)')
    parser.add_argument('--test_mc_iters', default=0, type=int,
                        help='Number of MC iters when testing (when using MC inference)')

    parser.add_argument('--init_params',
                        default=["{\"bias_type\":", "\"xavier\",", "\"conv_type\":", "\"xavier\",",
                                "\"bn_init\":", "\"01\"}"], type=str, nargs='*', help='Initialization parameters')

    parser.add_argument('--desc', default="", type=str, nargs='*',
                        help='Desc file content')

    parser.add_argument('--bw_to_rgb', default=False, action='store_true',
                        help='Convert black&white (e.g. MNIST) images to RGB format')

    parser.add_argument('--permuted_offset', default=False, action='store_true',
                        help='Use offset for permuted mnist experiment')
    parser.add_argument('--labels_trick', default=False, action='store_true',
                        help='Use labels trick (train only the heads of current batch labels)')

    parser.add_argument('--num_of_permutations', default=9, type=int,
                        help='Number of permutations (in addition to the original MNIST) ' 
                            'when using Permuted MNIST dataset [default: 9]')
    parser.add_argument('--iterations_per_virtual_epc', default=468, type=int,
                        help='When using continuous dataset, number of iterations per epoch (in continuous mode, '
                            'epoch is not defined)')
    
    #Because in continous setting there is no concept of a single task,so we bring in a concept of virtual iterations per epoch to relate this with the concept of epoch

    parser.add_argument('--separate_labels_space', default=False, action='store_true',
                        help='Use separate label space for each task')

    parser.add_argument('--permute_seed', type=int,
                        help='Seed for creating the permutations.')
    
    parser.add_argument('--federated_learning',default=False,action='store_true',help='To enable federated learning(client-server style training)')

    parser.add_argument('--n_clients',default = 5,type = int,help='Used to mention number of clients in federated learning setting')

    parser.add_argument('--num_aggs_per_task', default=1, type=int,
                        help='Number of aggregations happen in federated training(default 10)')

    parser.add_argument('--grad_clip',default=False,action = 'store_true',help='Indicates whether gradient clipping is being used')
    parser.add_argument('--max_grad_norm',default = 1.0 ,type = float, help='Used when grag clip is true')
    

    parser.add_argument('--non_iid_split', default=False, action='store_true',
                        help='Splits data in non_iid_fashion')
    
    parser.add_argument('--alpha', type=float, default = 1.0, 
                        help='degree_of_non_iid_dirichlet_parameter') # if not provided -> iid
    
    parser.add_argument('--alpha_mg',type=float,default=0.5,help = 'Mixture of gaussians parameter')
    
    args = parser.parse_args()

    return args