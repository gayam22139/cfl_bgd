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
import copy
from datetime import datetime
import pytz


#Arguments have been moved to a different file named argparser.py
args = get_args()

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


set_seed(args.permute_seed, fully_deterministic=False)
for p_idx in range(args.num_of_permutations):
    input_size = 32 * 32
    if "padded" not in args.dataset:
        input_size = 28 * 28
    permutation = list(range(input_size))
    random.shuffle(permutation)
    all_permutation.append(permutation)

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

IST = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(IST).strftime("%H:%M_%d-%m-%Y")


if args.grad_clip:
    if args.optimizer == 'sgd':
        args.results_dir = f"perm_mnist_{args.num_of_permutations + 1}_tasks_{args.num_epochs}_epochs_{args.lr}_lr_{args.optimizer}_optim__{args.contpermuted_beta}_beta_with_grad_clip_{current_time}"
    if args.optimizer == 'bgd':
        args.results_dir = f"perm_mnist_{args.num_of_permutations + 1}_tasks_{args.num_epochs}_epochs_{args.mean_eta}_mean_eta__{args.optimizer}_optim_{args.contpermuted_beta}_beta_with_grad_clip_{current_time}"
else:
    if args.optimizer == 'sgd':
        args.results_dir = f"perm_mnist_{args.num_of_permutations + 1}_tasks_{args.num_epochs}_epochs_{args.lr}_lr_{args.optimizer}_optim_{args.contpermuted_beta}_beta_{current_time}"
    if args.optimizer == 'bgd':
        args.results_dir = f"perm_mnist_{args.num_of_permutations + 1}_tasks_{args.num_epochs}_epochs_{args.mean_eta}_mean_eta_{args.optimizer}_optim_{args.contpermuted_beta}_beta_{current_time}"

save_path = os.path.join("./logs", str(args.results_dir) + "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)

args.logname = f"continous_permuted_mnist_{args.num_of_permutations + 1}_tasks"
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


def agg_client_models(client_models,client_optimizers):

    # server_model = client_models[0].named_parameters()

    # print(dir(client_models[0]))
    # print(dir(client_optimizers[0]))

    #print("-------OPTIMIZER FROM HERE-----")

    model_params = {}

    total_clients = len(client_optimizers.keys())

    for client_id in client_optimizers.keys():
    
        for layer_id,layer in enumerate(client_optimizers[client_id].param_groups):

            if client_id == 0:
                model_params[layer_id] = {}
                model_params[layer_id]['mean_param'] = torch.div(layer['mean_param'],total_clients)
                model_params[layer_id]['std_param'] = torch.div(layer['std_param'],total_clients)
            
            else: 
                model_params[layer_id]['mean_param'].add_(torch.div(layer['mean_param'],total_clients))
                model_params[layer_id]['std_param'].add_(torch.div(layer['std_param'],total_clients))


    for layer_id in model_params.keys():
        model_params[layer_id]['eps'] = torch.normal(torch.zeros_like(model_params[layer_id]['std_param']), 1)
        model_params[layer_id]['params'] = model_params[layer_id]['mean_param'].add(model_params[layer_id]['eps'].mul(model_params[layer_id]['std_param']))


    model_params_lst = [layer_weights['params'] for layer_id,layer_weights in model_params.items()]

    #print(model_params_lst,len(model_params_lst))


    agg_model = copy.deepcopy(client_models[0])

    agg_model_state_dict = {}


    # print(client_models[0].state_dict())
    # print("Length of model state dict is ",len(client_models[0].state_dict()))
    # print(dir(client_models[0]))

    layer_id = 0
    for layer in agg_model.state_dict().keys():
        # print("Model stats")
        # print(type(client_models[0].state_dict()[layer]))
        # print(client_models[0].state_dict()[layer].shape) 

        agg_model_state_dict[layer] = model_params_lst[layer_id]
        layer_id+=1

        # print("Our lst stats")
        # print(type(model_params_lst[layer_id]))
        # print(model_params_lst[layer_id].shape)
        # layer_id+=1
    
    # print(agg_model_state_dict)

    # print(agg_model)
    agg_model.load_state_dict(agg_model_state_dict)

    return agg_model 


def agg_client_models_sgd(client_models, client_optimizers):

    total_clients = len(client_optimizers.keys())
    agg_model_params = {} # layer_ids, layer_weights

    for client_id in client_optimizers.keys():

        for layer_id, layer in enumerate(client_optimizers[client_id].param_groups):

            if client_id == 0:
                agg_model_params[layer['name']] = torch.div(layer['params'][0], total_clients)
            
            else:
                agg_model_params[layer['name']].add_(torch.div(layer['params'][0], total_clients))

    #print(model_params_lst,len(model_params_lst))
    # breakpoint()

    agg_model = copy.deepcopy(client_models[0])
    # print("Length of model state dict is ",len(client_models[0].state_dict()))
    agg_model.load_state_dict(agg_model_params)

    return agg_model 

def test_agg_model(server_model,test_loaders):
 
    criterion = nn.CrossEntropyLoss()

    test_accuracies = []
    test_losses = []

    with torch.no_grad():
        server_model.eval()
        for test_loader in test_loaders:
            total_loss = 0
            accuracy = 0
            total_batches = 0
            total = 0
            correct = 0

            for data in test_loader:
                inputs,labels = data
                if torch.cuda.is_available():
                    inputs,labels = inputs.cuda(),labels.cuda()
                
                outputs = server_model(inputs)
                loss = criterion(outputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                total_batches+=1
                
            accuracy += correct / total
            total_loss = total_loss/total_batches
            test_accuracies.append(accuracy)
            test_losses.append(total_loss)

    avg_acc = sum(test_accuracies)/len(test_accuracies)
    avg_loss = sum(test_losses)/len(test_losses)
    
    logger.info(f"Task wise accuracies are {test_accuracies}")
    return round(avg_acc * 100,3),round(avg_loss,3)


# Dataset

if args.federated_learning:
    client_train_loaders, test_loaders = utils.datasets.__dict__[args.dataset](batch_size=args.batch_size,
                                                                        num_workers=args.num_workers,
                                                                        permutations=all_permutation,
                                                                        separate_labels_space=args.separate_labels_space,
                                                                        num_epochs=args.num_epochs,
                                                                        iterations_per_virtual_epc=
                                                                        args.iterations_per_virtual_epc,
                                                                        contpermuted_beta=args.contpermuted_beta,
                                                                        logger=logger,federated_learning = args.federated_learning,n_clients = args.n_clients)


    #Modify below code as per Federated requirements

    #Probes manager

    server_model = None
    client_models = {client_id:None for client_id in range(args.n_clients)}
    client_probes_managers = {client_id:None for client_id in range(args.n_clients)}
    client_trainers = {client_id:None for client_id in range(args.n_clients)}
    client_optimizers = {client_id:None for client_id in range(args.n_clients)}
    client_max_epoch_counter = {client_id:1 for client_id in range(args.n_clients)}

    total_rounds = args.num_of_permutations + 1

    optimizer_model = optimizers_lib.__dict__[args.optimizer]
    if args.optimizer == 'bgd':
        optimizer_params = dict({"logger": logger,
                                        "mean_eta": args.mean_eta,
                                        "std_init": args.std_init,
                                        "mc_iters": args.train_mc_iters}, **literal_eval(" ".join(args.optimizer_params)))
        
    if args.optimizer == 'sgd':
        optimizer_params = dict({"logger": logger,
                                        "momentum": args.momentum,
                                        "lr": args.lr,
                                        "weight_decay": args.weight_decay}, **literal_eval(" ".join(args.optimizer_params)))

    probes_manager = ProbesManager()
    server_model = models.__dict__[args.nn_arch](probes_manager=probes_manager)

    if torch.cuda.is_available():
        server_model = torch.nn.DataParallel(server_model).cuda()
        logger.info("Transformed model to CUDA")

    criterion = nn.CrossEntropyLoss()

    init_params = {"logger": logger}
    if args.init_params != "":
        init_params = dict(init_params, **literal_eval(" ".join(args.init_params)))

    init_model(get_model(server_model), **init_params)

    # optimizer model
    optimizer = optimizer_model(server_model, probes_manager=probes_manager, **optimizer_params)

    avg_test_accuracies = []
    avg_test_losses = []
    
    for round_no in range(total_rounds):

        for client_id in range(args.n_clients):
            # Model
            if client_models[client_id] == None and round_no == 0:
                # probes_manager = ProbesManager()
                # model = models.__dict__[args.nn_arch](probes_manager=probes_manager)

                # if torch.cuda.is_available():
                #     model = torch.nn.DataParallel(model).cuda()
                #     logger.info("Transformed model to CUDA")

                # criterion = nn.CrossEntropyLoss()

                # init_params = {"logger": logger}
                # if args.init_params != "":
                #     init_params = dict(init_params, **literal_eval(" ".join(args.init_params)))

                # init_model(get_model(model), **init_params)

                # # optimizer model
                # optimizer = optimizer_model(model, probes_manager=probes_manager, **optimizer_params)

            # print(dir(optimizer))
            # print(optimizer.param_groups)
            # print(optimizer.param_groups.keys())    
                current_client_trainer = NNTrainer(train_loader=[client_train_loaders[client_id]], test_loader=test_loaders,
                                    criterion=criterion, net=copy.deepcopy(server_model), logger=logger, probes_manager=copy.deepcopy(probes_manager),
                                    std_init=args.std_init, mean_eta=args.mean_eta, train_mc_iters=args.train_mc_iters,
                                    test_mc_iters=args.test_mc_iters, committee_size=args.committee_size, batch_size=args.batch_size,
                                    inference_methods=inference_methods,
                                    pruning_percents=args.pruning_percents,
                                    bw_to_rgb=args.bw_to_rgb,
                                    labels_trick=args.labels_trick,
                                    test_freq=args.test_freq,
                                    optimizer=copy.deepcopy(optimizer))

                current_client_trainer.net = copy.deepcopy(server_model)
                #current_client_trainer.net = copy.deepcopy(server_model)
                current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, probes_manager=current_client_trainer.probes_manager, **optimizer_params)

            #Update the model in the trainer object
            else:
                current_client_trainer = client_trainers[client_id]
                current_client_trainer.net = copy.deepcopy(server_model)
                #current_client_trainer.net = copy.deepcopy(server_model)
                current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, probes_manager=current_client_trainer.probes_manager, **optimizer_params)
                
            """In federated setting the number of epochs is always 1,as we consider all iterations as 
                one big epoch(flattened all iterations over all epochs as a single epoch)"""

            if args.grad_clip:
                logger.info("Gradient clipping with max_norm being done")
            
            
            current_client_trainer.train_epochs(verbose_freq=500, max_epoch=client_max_epoch_counter[client_id],
                                permanent_prune_on_epoch=args.permanent_prune_on_epoch,
                                permanent_prune_on_epoch_percent=args.permanent_prune_on_epoch_percent,federated_learning=args.federated_learning,client_id = client_id,round = round_no,grad_clip = args.grad_clip)
            
            # print(dir(trainer))
            # print(dir(trainer.train_loader[0]))
            # print(trainer.train_loader[0].sampler)
            # print(dir(trainer.train_loader[0].sampler))
            # print(trainer.train_loader[0].sampler.current_iteration)
            # exit()

            client_trainers[client_id] = current_client_trainer
            client_models[client_id] = current_client_trainer.net
            client_optimizers[client_id] = current_client_trainer.optimizer

            client_max_epoch_counter[client_id]+=1

            print(f"Round - {round_no+1},Client - {client_id+1}")

            

            print(f"{current_client_trainer.train_loader[0].sampler.current_iteration},",
                  f"{current_client_trainer.train_loader[0].sampler.current_round_start_iter},",
                  f"{current_client_trainer.train_loader[0].sampler.current_round_end_iter}")
        
        if args.optimizer == 'sgd':
            server_model = agg_client_models_sgd(client_models, client_optimizers)

        if args.optimizer == 'bgd':
            server_model = agg_client_models(client_models, client_optimizers)

        # Save the aggregated model
            
        agg_model_dir = os.path.join(save_path, 'agg_models')
        if not os.path.exists(agg_model_dir):
            os.makedirs(agg_model_dir)
        model_save_path = os.path.join(agg_model_dir, f'aggregated_model_round_{round_no+1}.pth')
        torch.save(server_model.state_dict(), model_save_path)

        total = 0
        with torch.no_grad():
            for layer in server_model.parameters():
                total+=torch.sum(layer)


        avg_test_accuracies.append(test_agg_model(server_model,test_loaders)[0])
        avg_test_losses.append(test_agg_model(server_model,test_loaders)[1])


        logger.info(f"Value of server model at round {round_no+1} is {total.item()}")
        logger.info(f"Aggregated model avg acc and avg loss - {test_agg_model(server_model,test_loaders)}")
        
        logger.info(f"Round - {round_no+1} complete")

    if args.optimizer == 'sgd':
        get_accuracy_loss_plot(avg_test_accuracies=avg_test_accuracies, avg_test_losses= avg_test_losses, results_dir= save_path, optimizer = args.optimizer, lr = args.lr)
    
    if args.optimizer == 'bgd':
        get_accuracy_loss_plot(avg_test_accuracies=avg_test_accuracies, avg_test_losses= avg_test_losses, results_dir= save_path, optimizer = args.optimizer, lr = args.mean_eta)

    
    logger.info(f"Avg Test Accuracies : {avg_test_accuracies}")
    logger.info(f"Avg Test Losses : {avg_test_losses}")
    logger.info("Done - Federated Learning Setup")

else:
    train_loaders, test_loaders = utils.datasets.__dict__[args.dataset](batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    permutations=all_permutation,
                                                                    separate_labels_space=args.separate_labels_space,
                                                                    num_epochs=args.num_epochs,
                                                                    iterations_per_virtual_epc=
                                                                    args.iterations_per_virtual_epc,
                                                                    contpermuted_beta=args.contpermuted_beta,
                                                                    logger=logger)

    # Probes manager
    probes_manager = ProbesManager()

    # Model
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

    # print(trainer.optimizer.param_groups)


    print("Done")