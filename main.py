from utils.utils import *
from utils.logging_utils import Logger
import utils.datasets
from nn_utils.NNTrainer import NNTrainer
from probes_lib.top import *
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from argparser import get_args
import models
from time import time
import os
import socket
from ast import literal_eval
from nn_utils.init_utils import init_model
import optimizers_lib


import pdb

#Arguments have been moved to a different file named argparser.py
args = get_args()

# print("String args is ",str(args),"\n\n\n")

###########################################################################
# Verify arguments
###########################################################################
inference_methods = set()
if args.inference_committee:
    assert (args.committee_size > 0)
    inference_methods.add("committee")
if args.inference_mc:
    assert (args.test_mc_iters > 0)
    inference_methods.add("test_mc")
if args.inference_map:
    inference_methods.add("map")
if args.inference_aggsoftmax:
    inference_methods.add("agg_softmax")
if args.inference_initstd:
    inference_methods.add("init_std")

assert(len(inference_methods) > 0)

if args.optimizer != "bgd":
    assert args.train_mc_iters == 1, "Monte Carlo iterations are for BGD optimizer only"
    assert len(inference_methods) == 1 and "map" in inference_methods, "When not using BGD, must use MAP for inference"

###########################################################################
# CUDA and seeds
###########################################################################

# Create permuations
all_permutation = []
if not args.permute_seed:
    args.permute_seed = int(time()*10000) % (2**31)

if args.n_tasks:
    args.num_of_permutations = args.n_tasks - 1


# print(args.num_of_permutations)


set_seed(args.permute_seed, fully_deterministic=False)
for p_idx in range(args.num_of_permutations):
    input_size = 32 * 32
    if "padded" not in args.dataset:
        input_size = 28 * 28
    permutation = list(range(input_size))
    random.shuffle(permutation)
    all_permutation.append(permutation)


# print("All permutations is ",len(all_permutation))

# print("A sample permuation ",all_permutation[3])
# print(len(all_permutation[3]))

#All permutations is a list of lists -> where the size of outer list is number of permuations(received from command line) and each permuation is a list of shuffled(permuted) indices(an arrangement of 784 indices)

# Set seed
if not args.seed:
    args.seed = int(time()*10000) % (2**31)

set_seed(args.seed, fully_deterministic=False)

if torch.cuda.is_available():
    cudnn.benchmark = True


###########################################################################
# Logging
###########################################################################
# Create logger
save_path = os.path.join("./logs", str(args.results_dir) + "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(True, save_path + args.logname, True, True)

logger.info("Script args: " + str(args))

if args.desc != "":
    logger.create_desc_file(" ".join(args.desc))
else:
    logger.create_desc_file(str(args))


logger.info("Computer name: " + str(socket.gethostname()) + " with pytorch version: " + str(torch.__version__))

lastlogs_logger = Logger(add_timestamp=False, logfile_name="last_logs.txt", logfile_name_time_suffix=False,
                         print_to_screen=False)
lastlogs_logger.info(logger.get_log_basename() + " ")
lastlogs_logger = None


###########################################################################
# Model and training
###########################################################################

print("Dataset is ",args.dataset)

# pdb.set_trace()

# Dataset
train_loaders, test_loaders = utils.datasets.__dict__[args.dataset](batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    permutations=all_permutation,
                                                                    separate_labels_space=args.separate_labels_space,
                                                                    num_epochs=args.num_epochs,
                                                                    iterations_per_virtual_epc=
                                                                    args.iterations_per_virtual_epc,
                                                                    contpermuted_beta=args.contpermuted_beta,
                                                                    logger=logger)




#Sample trainloader

# print(len(train_loaders))
# print(train_loaders[0])

sample_batch = next(iter(train_loaders[0]))

input,labels = sample_batch

# print(input.shape)

# with open("sample_input.txt","w") as f:
#     f.write(str(input))


# print("IN main program",len(test_loaders))
# print("train loaders ",len(train_loaders))

# print(train_loaders[0])

# Probes manager
probes_manager = ProbesManager()

# Model
if args.n_tasks and args.n_classes:
    model = models.__dict__[args.nn_arch](args.n_classes,args.n_tasks,probes_manager=probes_manager)

else:
    model = models.__dict__[args.nn_arch](probes_manager=probes_manager)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Transformed model to CUDA")

criterion = nn.CrossEntropyLoss()

init_params = {"logger": logger}
if args.init_params != "":
    init_params = dict(init_params, **literal_eval(" ".join(args.init_params)))

init_model(get_model(model), **init_params)

# optimizer model
optimizer_model = optimizers_lib.__dict__[args.optimizer]
optimizer_params = dict({"logger": logger,
                         "mean_eta": args.mean_eta,
                         "std_init": args.std_init,
                         "mc_iters": args.train_mc_iters}, **literal_eval(" ".join(args.optimizer_params)))
optimizer = optimizer_model(model, probes_manager=probes_manager, **optimizer_params)

trainer = NNTrainer(train_loader=train_loaders, test_loader=test_loaders,
                    criterion=criterion, net=model, logger=logger, probes_manager=probes_manager,
                    std_init=args.std_init, mean_eta=args.mean_eta, train_mc_iters=args.train_mc_iters,
                    test_mc_iters=args.test_mc_iters, committee_size=args.committee_size, batch_size=args.batch_size,
                    inference_methods=inference_methods,
                    pruning_percents=args.pruning_percents,
                    bw_to_rgb=args.bw_to_rgb,
                    labels_trick=args.labels_trick,
                    test_freq=args.test_freq,
                    optimizer=optimizer)


trainer.train_epochs(verbose_freq=100, max_epoch=args.num_epochs,
                     permanent_prune_on_epoch=args.permanent_prune_on_epoch,
                     permanent_prune_on_epoch_percent=args.permanent_prune_on_epoch_percent)


print("Done")