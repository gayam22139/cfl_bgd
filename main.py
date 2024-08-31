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
# import wandb
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

if args.optimizer != "bgd_new_update" and args.optimizer != "bgd":
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

# split_classes_list
classes_lst = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ]


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


if args.optimizer == 'sgd':
    args.results_dir = f"{args.dataset}_{args.num_of_permutations + 1 if args.dataset=='ds_cont_permuted_mnist' else len(classes_lst)}_tasks_{args.num_epochs}_epochs_alpha_{args.alpha}_data_type_{'non_iid' if args.non_iid_split else 'iid'}_{args.lr}_lr_{args.optimizer}_optim_{args.contpermuted_beta}_beta{'_with_grad_clip' if args.grad_clip else '_'}_{current_time}"
if args.optimizer == 'bgd_new_update'or args.optimizer == 'bgd':
    args.results_dir = f"{args.dataset}_{args.num_of_permutations + 1 if args.dataset=='ds_cont_permuted_mnist' else len(classes_lst)}_tasks_{args.num_epochs}_epochs_alpha_{args.alpha}_data_type_{'non_iid' if args.non_iid_split else 'iid'}_{args.mean_eta}_mean_eta__{args.optimizer}_optim_{args.contpermuted_beta}_beta{'_with_grad_clip' if args.grad_clip else '_'}_{current_time}"



save_path = os.path.join("./logs", str(args.results_dir) + "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)


if not os.path.exists('all_experiments_results'):
    os.makedirs('all_experiments_results')

# Write the initial message to the file before entering the loop
with open(f'all_experiments_results/{args.results_dir}.txt', 'w') as f:
    f.write(f"{'*'*100}\n")
    f.write(f"Writing results to {args.results_dir}\n")

if args.dataset == 'ds_cont_permuted_mnist':
    args.logname = f"continous_permuted_mnist_{args.num_of_permutations + 1}_tasks"

if args.dataset == 'ds_cont_split_mnist':
    args.logname = f"continous_split_mnist_{len(classes_lst)}_tasks"

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


''' prev aggregation '''


def agg_client_models_avg(client_models,client_optimizers):

    #server_model = client_models[0].named_parameters()

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

    # Rename keys to g_mean_param and g_std_param
    

    # print(agg_model)
    agg_model.load_state_dict(agg_model_state_dict)

    new_model_params = {}
    for layer_id, layer_weights in model_params.items():
        new_model_params[layer_id] = {
            'g_mean_param': layer_weights['mean_param'],
            'g_std_param': layer_weights['std_param']
        }

    return agg_model, new_model_params # model_params # we are saying aggregated model as model_


def agg_client_models_new(client_models, client_optimizers):

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
                model_params[layer_id]['mean_param'] = torch.div(layer['mean_param'], total_clients)
    
            else: 
                model_params[layer_id]['mean_param'].add_(torch.div(layer['mean_param'], total_clients))
       
    for client_id in client_optimizers.keys():

        for layer_id,layer in enumerate(client_optimizers[client_id].param_groups):

            current_client_variance = torch.square(layer['std_param'])
            current_client_mean_square = torch.square(layer['mean_param'])
            server_mean_param_square = torch.square(model_params[layer_id]['mean_param'])
            
            if client_id == 0:
                model_params[layer_id]['variance_param'] = torch.div(torch.sub(torch.add(current_client_variance, current_client_mean_square), server_mean_param_square ), total_clients)

            else :
                model_params[layer_id]['variance_param'].add_(torch.div(torch.sub(torch.add(current_client_variance, current_client_mean_square), server_mean_param_square ), total_clients))

    

    for layer_id,layer in enumerate(client_optimizers[client_id].param_groups):

        model_params[layer_id]['std_param'] = torch.sqrt(model_params[layer_id]['variance_param'])

       

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

    new_model_params = {}
    for layer_id, layer_weights in model_params.items():
        new_model_params[layer_id] = {
            'g_mean_param': layer_weights['mean_param'],
            'g_std_param': layer_weights['std_param']
        }

    return agg_model, new_model_params


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

def test_agg_model(server_model,test_loaders, round_no):
 
    criterion = nn.CrossEntropyLoss()


    test_accuracies = []
    test_losses = []

    task_no = (round_no) // (args.num_aggs_per_task) # always be from 0 - 9 || 0-4 rounds (task_no 0) || 5-9 rounds (task_no 1)

    with torch.no_grad():
        server_model.eval()
        for test_loader in test_loaders[:task_no+1]: # [:round_no+1] || 
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
            accuracy = accuracy * 100
            test_accuracies.append(accuracy)
            test_losses.append(total_loss)

    avg_acc = sum(test_accuracies)/len(test_accuracies)
    avg_loss = sum(test_losses)/len(test_losses)
  
    
     # and task_wise_accuracies_lst {test_accuracies}")
    
    rounded_test_accuracies = [round(num, 3) for num in test_accuracies]
    rounded_test_losses = [round(num, 3) for num in test_losses]

    logger.info(f"Task wise accuracies after round {round_no+1} are {rounded_test_accuracies}")
    
    return round(avg_acc,3), round(avg_loss,3), rounded_test_accuracies, rounded_test_losses 


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
                                                                        logger=logger,federated_learning = args.federated_learning,
                                                                        n_clients = args.n_clients, non_iid_split=args.non_iid_split, 
                                                                        num_aggs_per_task = args.num_aggs_per_task, 
                                                                        classes_lst = classes_lst,
                                                                        alpha = args.alpha)


    #Modify below code as per Federated requirements

    #Probes manager

    server_model = None
    client_models = {client_id:None for client_id in range(args.n_clients)}
    client_probes_managers = {client_id:None for client_id in range(args.n_clients)}
    client_trainers = {client_id:None for client_id in range(args.n_clients)}
    client_optimizers = {client_id:None for client_id in range(args.n_clients)}
    client_max_epoch_counter = {client_id:1 for client_id in range(args.n_clients)}

    ''' Number of  aggregations per one task multiplied by num_of_tasks gives us total_fl_rounds '''
    
    if args.dataset == 'ds_cont_permuted_mnist':
        total_rounds = (args.num_of_permutations + 1) * (args.num_aggs_per_task)

    if args.dataset == 'ds_cont_split_mnist':
        total_rounds = (len(classes_lst)) * (args.num_aggs_per_task)

    
    

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

    if args.optimizer == 'bgd_new_update':
        optimizer_params = dict({"logger": logger,
                                        "mean_eta": args.mean_eta,
                                        "std_init": args.std_init,
                                        "mc_iters": args.train_mc_iters,"alpha_mg":args.alpha_mg}, **literal_eval(" ".join(args.optimizer_params)))
    

    probes_manager = ProbesManager()
    server_model = models.__dict__[args.nn_arch](probes_manager=probes_manager)

    if torch.cuda.is_available():
        server_model = torch.nn.DataParallel(server_model).cuda()
        logger.info("Transformed model to CUDA")

    # breakpoint()

    criterion = nn.CrossEntropyLoss()

    init_params = {"logger": logger}
    if args.init_params != "":
        init_params = dict(init_params, **literal_eval(" ".join(args.init_params)))

    init_model(get_model(server_model), **init_params)

     # mean --> params
    server_model_params  = {}

    # print(server_model.module)
    
    for layer_name, param in server_model.named_parameters():
        # print(layer_name)
        server_model_params[layer_name] = {}

        server_model_params[layer_name]['g_mean_param'] = param
        server_model_params[layer_name]['g_std_param'] = torch.full_like(param,args.std_init) 

    # optimizer model
    if args.optimizer == 'bgd_new_update':
        optimizer = optimizer_model(server_model, server_model_params,probes_manager=probes_manager, **optimizer_params) 
    else:
        optimizer = optimizer_model(server_model,probes_manager=probes_manager, **optimizer_params)

    avg_test_accuracies = []
    avg_test_losses = []
    task_wise_accuracies_every_round = []
    task_wise_losses_every_round = []

    ''' Initialize wandb '''

    # wandb.init(project="bgd_fl_new_agg_strategy_v2", config={
    #     "learning_rate": args.mean_eta,
    #     "optimizer": args.optimizer,
    #     "total_rounds": total_rounds,
    #     "n_clients": args.n_clients,
    #     "max_grad_norm" : args.max_grad_norm
    # }) 

    for round_no in range(total_rounds): 
        avg_acc, avg_loss = 0, 0
        for client_id in range(args.n_clients):
            # Model
            if client_models[client_id] == None and round_no == 0:

                current_client_trainer = NNTrainer(train_loader=[client_train_loaders[client_id]], test_loader=test_loaders,
                                    criterion=criterion, net=copy.deepcopy(server_model), logger=logger, probes_manager=copy.deepcopy(probes_manager),
                                    std_init=args.std_init, mean_eta=args.mean_eta, train_mc_iters=args.train_mc_iters,
                                    test_mc_iters=args.test_mc_iters, committee_size=args.committee_size, batch_size=args.batch_size,
                                    inference_methods=inference_methods,
                                    pruning_percents=args.pruning_percents,
                                    bw_to_rgb=args.bw_to_rgb,
                                    labels_trick=args.labels_trick,
                                    test_freq=args.test_freq,
                                    optimizer=copy.deepcopy(optimizer),
                                    max_grad_norm = args.max_grad_norm
                                    ) 

                current_client_trainer.net = copy.deepcopy(server_model)

                #current_client_trainer.net = copy.deepcopy(server_model)
                
                if args.optimizer == 'bgd_new_update':
                    current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, server_model_params, probes_manager=current_client_trainer.probes_manager, **optimizer_params)

                else:
                    current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, probes_manager=current_client_trainer.probes_manager, **optimizer_params)

            #Update the model in the trainer object
            else:
                current_client_trainer = client_trainers[client_id]
                current_client_trainer.net = copy.deepcopy(server_model)
                #current_client_trainer.net = copy.deepcopy(server_model)

                if args.optimizer == 'bgd_new_update':
                    current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, server_model_params,  probes_manager=current_client_trainer.probes_manager, **optimizer_params)

                else:
                    current_client_trainer.optimizer = optimizer_model(current_client_trainer.net, probes_manager=current_client_trainer.probes_manager, **optimizer_params)

                
            """In federated setting the number of epochs is always 1,as we consider all iterations as 
                one big epoch(flattened all iterations over all epochs as a single epoch)"""

            if args.grad_clip:
                logger.info("Gradient clipping with max_norm being done")
            
            # if round_no > 0:
            current_client_trainer.train_epochs(verbose_freq=500, max_epoch=client_max_epoch_counter[client_id],
                                    permanent_prune_on_epoch=args.permanent_prune_on_epoch,
                                    permanent_prune_on_epoch_percent=args.permanent_prune_on_epoch_percent,federated_learning=args.federated_learning,client_id = client_id,round = round_no,grad_clip = args.grad_clip)
                
            # print(dir(trainer))
            # print(dir(trainer.train_loader[0]))
            # print(trainer.train_loader[0].sampler)
            # print(dir(trainer.train_loader[0].sampler))
            # print(trainer.train_loader[0].sampler.current_iteration)
            # exit()

            # Log gradients and weights to wandb
            # for name, param in current_client_trainer.net.named_parameters():
            #     if param.grad is not None:
            #         wandb.log({
            #             "round": round_no + 1,
            #             "client": client_id + 1,
            #             f"client_{client_id}/gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
            #             f"client_{client_id}/weights/{name}": wandb.Histogram(param.data.cpu().numpy())
            #         })

            # wandb.watch(current_client_trainer.net, log = 'all')
            #  Watch the model 
            # wandb.watch(current_client_trainer.net, log='all', log_freq = 1)

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

        if args.optimizer == 'bgd_new_update' or args.optimizer == 'bgd':
            # new aggregation
            server_model, server_model_params = agg_client_models_avg(client_models, client_optimizers)
            #server_model, server_model_params = agg_client_models_new(client_models, client_optimizers)

            # # avg aggregation
            # server_model = agg_client_models_avg(client_models, client_optimizers)

        


        # # Save the aggregated model
            
        # agg_model_dir = os.path.join(save_path, 'agg_models')
        # if not os.path.exists(agg_model_dir):
        #     os.makedirs(agg_model_dir)
        # model_save_path = os.path.join(agg_model_dir, f'aggregated_model_round_{round_no+1}.pth')
        # torch.save(server_model.state_dict(), model_save_path)

        total = 0
        with torch.no_grad():
            for layer in server_model.parameters():
                total+=torch.sum(layer)

        round_avg_acc, round_avg_loss, task_wise_test_acc, task_wise_test_loss = test_agg_model(server_model,test_loaders, round_no)
        avg_test_accuracies.append(round_avg_acc)
        avg_test_losses.append(round_avg_loss)
        task_wise_accuracies_every_round.append(task_wise_test_acc)
        task_wise_losses_every_round.append(task_wise_test_loss)

        # wandb.log({
        #     "round_avg_acc": round_avg_acc,
        #     "round_avg_loss": round_avg_loss
        # })

        logger.info(f"Value of server model at round {round_no+1} is {total.item()}")
        logger.info(f"Aggregated model avg acc and avg loss - {round_avg_acc, round_avg_loss}")
        
        logger.info(f"Round - {round_no+1} complete")

    with open(f'all_experiments_results/{args.results_dir}.txt', 'a') as f:
        f.write(f"Task-wise accuracies {args.optimizer}_optim after all rounds are {task_wise_accuracies_every_round}\n")
        f.write(f"Task-wise losses {args.optimizer}_optim after all rouds are {task_wise_losses_every_round}\n")
        f.write(f"Average accuracies {args.optimizer}_optim after all rouds are {avg_test_accuracies}\n")
        f.write(f"Average losses {args.optimizer}_optim after all rouds are {avg_test_losses}\n")


    if args.optimizer == 'sgd':
        get_accuracy_loss_plot(avg_test_accuracies=avg_test_accuracies, avg_test_losses= avg_test_losses, results_dir= save_path, optimizer = args.optimizer, lr = args.lr)
    
    if args.optimizer == 'bgd':
        get_accuracy_loss_plot(avg_test_accuracies=avg_test_accuracies, avg_test_losses= avg_test_losses, results_dir= save_path, optimizer = args.optimizer, lr = args.mean_eta)

    
    logger.info(f"Avg Test Accuracies : {avg_test_accuracies}")
    logger.info(f"Avg Test Losses : {avg_test_losses}")
    logger.info("Done - Federated Learning Setup")
    # wandb.finish()

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