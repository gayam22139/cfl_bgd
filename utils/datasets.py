import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
import os
import codecs
from torch.distributions.categorical import Categorical
import torch.utils.data as data
from PIL import Image
import errno
import copy
import time


def _reduce_class(set, classes, train, preserve_label_space=True):
    if classes is None:
        return

    new_class_idx = {}
    for c in classes:
        new_class_idx[c] = new_class_idx.__len__()

    new_data = []
    new_labels = []
    if train:
        all_data = set.train_data
        labels = set.train_labels
    else:
        all_data = set.test_data
        labels = set.test_labels

    for data, label in zip(all_data, labels):
        if type(label) == int:
            label_val = label
        else:
            label_val = label.item()
        if label_val in classes:
            new_data.append(data)
            if preserve_label_space:
                new_labels += [label_val]
            else:
                new_labels += [new_class_idx[label_val]]
    if type(new_data[0]) == np.ndarray:
        new_data = np.array(new_data)
    elif type(new_data[0]) == torch.Tensor:
        new_data = torch.stack(new_data)
    else:
        assert False, "Reduce class not supported"
    if train:
        set.train_data = new_data
        set.train_labels = new_labels
    else:
        set.test_data = new_data
        set.test_labels = new_labels


class Permutation(torch.utils.data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, permute_idx, target_offset):
        super(Permutation,self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx
        self.target_offset = target_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target = target + self.target_offset
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target


def iid_partition(dataset, clients):
  """
  I.I.D paritioning of data over clients
  Shuffle the data
  Split it between clients
  
  params:
    - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
    - clients (int): Number of Clients to split the data between

  returns:
    - Dictionary of image indexes for each client
  """
  x_train = dataset[0] 

  num_items_per_client = int(len(x_train)/clients)
  client_dict = {}
  image_idxs = [i for i in range(len(x_train))]

  for i in range(clients):
    client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
    image_idxs = list(set(image_idxs) - client_dict[i])

  return client_dict 

def non_iid_partition(dataset, clients, total_shards, shards_size, num_shards_per_client):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the clients will get X shards

    params:
        - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
        - clients (int): Number of Clients to split the data between
        - total_shards (int): Number of shards to partition the data in
        - shards_size (int): Size of each shard 
        - num_shards_per_client (int): Number of shards of size shards_size that each client receives

    returns:
        - Dictionary of image indexes for each client
    """
    x_train, y_train = dataset[0], dataset[1]

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
    idxs = np.arange(len(x_train))
    #data_labels = dataset.targets.numpy()

    # idxs = [0,1,2,......,59999],y_train = [0,3,6,4,7,3,9,...........]

    #label_idxs is 2x60,000 - matrix ,1st row is idxs and second row is y_train

    # sort the labels
    label_idxs = np.vstack((idxs, y_train))
    label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
    #after the above line label_idxs = [[0,....59999],[all_class0_idxs,all_class1_idxs,......all_class9_idxs]]
    idxs = label_idxs[0,:] 

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False)) # Randomly select shard_idx / shard_idxs
        shard_idxs = list(set(shard_idxs) - rand_set) # remove selected shard_idx from shard_idxs

        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
    
    return client_dict  




"""all_datasets - full_dataset,n_clients - number of clients to which this data has to be distributed"""

"""Two ways of dividing the combined dataset
-First way - Divide each task dataset among clients(using desired strategy - IID/Non-IID) one-by-one
-Second way - Divide the combined dataset(treat all datapoints as a single dataset) among clients.(Here,the risk 
associated is there can be a scenario where samples belonging to a particular task
 do not go to a particular client)(Can this cause gradient shift?)"""

def generate_client_datasets(tasks_datasets, non_iid_split, n_clients = 5, alpha = None):
    start_time = time.time()
    print("Client Indices Generation starts")

    n_tasks = len(tasks_datasets)
    
    clients_tasks_samples_indices = {client_id:[] for client_id in range(n_clients)}
    
    """client 0's tasks_samples_indices would be clients_tasks_samples_indices[0]
    
    Format of clients_tasks_samples_indices = {0:[[],[],[],[],....],1:[],2:[],....}
    
    """

    for task_id in range(n_tasks):
        dataset = tasks_datasets[task_id]

        dataset_x = [dataset[i][0] for i in range(len(dataset))]
        dataset_y = [dataset[i][1] for i in range(len(dataset))]    

        dataset = [dataset_x,dataset_y]

        
        if non_iid_split:
            print(f"Non-IID split is happening")
            train_size = len(dataset[1])
            client_dict = non_iid_partition(dataset, clients=n_clients, total_shards=n_clients, shards_size=train_size//n_clients, num_shards_per_client=1)

        else:
            print(f"IID split is happening")
            client_dict = iid_partition(dataset,n_clients)

        for client_id in client_dict.keys():
            client_dict[client_id] = torch.tensor([len(dataset_x)*task_id+index for index in client_dict[client_id]])
            clients_tasks_samples_indices[client_id].append(client_dict[client_id])

    end_time = time.time()

    print("Clients Indices Generation ends")
    print("Client Indices Generation took",end_time-start_time,"seconds")

    return clients_tasks_samples_indices

def _create_probabilites_over_iterations(total_iters,total_datasets,beta):
    tasks_probs_over_iterations = [_create_task_probs(total_iters, total_datasets, task_id,beta=beta)[0] for task_id in range(total_datasets)]
        
    normalize_probs = torch.zeros_like(tasks_probs_over_iterations[0])
    for probs in tasks_probs_over_iterations:
        normalize_probs.add_(probs)
    for probs in tasks_probs_over_iterations:
        probs.div_(normalize_probs)
    tasks_probs_over_iterations = torch.cat(tasks_probs_over_iterations).view(-1, tasks_probs_over_iterations[0].shape[0])
    tasks_probs_over_iterations_lst = []
    for col in range(tasks_probs_over_iterations.shape[1]):
        tasks_probs_over_iterations_lst.append(tasks_probs_over_iterations[:, col])
        
    return tasks_probs_over_iterations_lst


def get_classes_idx_list(full_mnist_dataset,cl_list):
    labels = full_mnist_dataset.targets
    idx_list = []
    for label_idx,label in enumerate(labels):
        if label in cl_list:
            idx_list.append(label_idx)
    
    return idx_list

def relabel_classes(mnist_subset):
    mnist_subset.dataset.targets = mnist_subset.dataset.targets % 2
    return mnist_subset

class DatasetsLoaders:
    def __init__(self, dataset, batch_size=4, num_workers=4, pin_memory=True, **kwargs):
        # print("kwargs in datasetloaders - ",kwargs)
        self.dataset_name = dataset
        self.valid_loader = None
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 4

        self.random_erasing = kwargs.get("random_erasing", False)
        self.reduce_classes = kwargs.get("reduce_classes", None)
        self.permute = kwargs.get("permute", False)
        self.target_offset = kwargs.get("target_offset", 0)

        self.federated_learning = kwargs.get("fl",False)
        self.n_clients = kwargs.get("n_clients",5)
        self.num_aggs_per_task = kwargs.get("num_aggs_per_task", 5)
        self.non_iid_split = kwargs.get("non_iid_split", False)

        pin_memory = pin_memory if torch.cuda.is_available() else False
        self.batch_size = batch_size
        cifar10_mean = (0.5, 0.5, 0.5)
        cifar10_std = (0.5, 0.5, 0.5)
        cifar100_mean = (0.5070, 0.4865, 0.4409)
        cifar100_std = (0.2673, 0.2564, 0.2761)
        mnist_mean = [33.318421449829934]
        mnist_std = [78.56749083061408]
        fashionmnist_mean = [73.14654541015625]
        fashionmnist_std = [89.8732681274414]

        

        if dataset == "CIFAR10":
            # CIFAR10:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 125.306918046875
            #                        train_set.train_data[:,:,:,1].mean() 122.95039414062499
            #                        train_set.train_data[:,:,:,2].mean() 113.86538318359375
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 62.993219278136884
            #                        train_set.train_data[:, :, :, 1].std() 62.088707640014213
            #                        train_set.train_data[:, :, :, 2].std() 66.704899640630913
            self.mean = cifar10_mean
            self.std = cifar10_std

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                          download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                         download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "CIFAR100":
            # CIFAR100:
            #   type               : uint8
            #   shape              : train_set.train_data.shape (50000, 32, 32, 3)
            #   test data shape    : (10000, 32, 32, 3)
            #   number of channels : 3
            #   Mean per channel   : train_set.train_data[:,:,:,0].mean() 129.304165605/255=0.5070
            #                        train_set.train_data[:,:,:,1].mean() 124.069962695/255=0.4865
            #                        train_set.train_data[:,:,:,2].mean() 112.434050059/255=0.4409
            #   Std per channel   :  train_set.train_data[:, :, :, 0].std() 68.1702428992/255=0.2673
            #                        train_set.train_data[:, :, :, 1].std() 65.3918080439/255=0.2564
            #                        train_set.train_data[:, :, :, 2].std() 70.418370188/255=0.2761

            self.mean = cifar100_mean
            self.std = cifar100_std
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(self.mean, self.std)])

            self.train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                           download=True, transform=transform)
            _reduce_class(self.train_set, self.reduce_classes, train=True,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                          download=True, transform=transform)
            _reduce_class(self.test_set, self.reduce_classes, train=False,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "MNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : 33.318421449829934
            #   Std per channel    : 78.56749083061408

            # Transforms
            self.mean = mnist_mean
            self.std = mnist_std
            if kwargs.get("pad_to_32", False):
                transform = transforms.Compose(
                    [transforms.Pad(2, fill=0, padding_mode='constant'),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.1000,), std=(0.2752,))])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor()])

            # Create train set
            self.train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                                        download=True, transform=transform)
            if kwargs.get("permutation", False):
                # Permute if permutation is provided
                self.train_set = Permutation(torchvision.datasets.MNIST(root='./data', train=True,
                                                                        download=True, transform=transform),
                                             kwargs.get("permutation", False), self.target_offset)
            # Reduce classes if necessary
            _reduce_class(self.train_set, self.reduce_classes, train=True,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            # Remap labels
            if kwargs.get("labels_remapping", False):
                labels_remapping = kwargs.get("labels_remapping", False)
                for lbl_idx in range(len(self.train_set.train_labels)):
                    self.train_set.train_labels[lbl_idx] = labels_remapping[self.train_set.train_labels[lbl_idx]]

            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            # Create test set
            self.test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                                       download=True, transform=transform)
            if kwargs.get("permutation", False):
                # Permute if permutation is provided
                self.test_set = Permutation(torchvision.datasets.MNIST(root='./data', train=False,
                                                                        download=True, transform=transform),
                                             kwargs.get("permutation", False), self.target_offset)
            # Reduce classes if necessary
            _reduce_class(self.test_set, self.reduce_classes, train=False,
                          preserve_label_space=kwargs.get("preserve_label_space"))
            # Remap labels
            if kwargs.get("labels_remapping", False):
                labels_remapping = kwargs.get("labels_remapping", False)
                for lbl_idx in range(len(self.test_set.test_labels)):
                    self.test_set.test_labels[lbl_idx] = labels_remapping[self.test_set.test_labels[lbl_idx]]

            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "FashionMNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : fm.train_data.type(torch.FloatTensor).mean() is 72.94035223214286
            #   Std per channel    : fm.train_data.type(torch.FloatTensor).std() is 90.0211833054075
            self.mean = fashionmnist_mean
            self.std = fashionmnist_std
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize(self.mean, self.std)])
            # transform = transforms.Compose(
            #     [transforms.ToTensor()])
            transform = transforms.Compose(
                [transforms.Pad(2),
                 transforms.ToTensor(),
                 transforms.Normalize((72.94035223214286 / 255,), (90.0211833054075 / 255,))])



            self.train_set = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=True,
                                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=False,
                                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "SVHN":
            # SVHN:
            #   type               : numpy.ndarray
            #   shape              : self.train_set.data.shape is (73257, 3, 32, 32)
            #   test data shape    : self.test_set.data.shape is (26032, 3, 32, 32)
            #   number of channels : 3
            #   Mean per channel   : sv.data.mean(axis=0).mean(axis=1).mean(axis=1) is array([111.60893668, 113.16127466, 120.56512767])
            #   Std per channel    : np.transpose(sv.data, (1, 0, 2, 3)).reshape(3,-1).std(axis=1) is array([50.49768174, 51.2589843 , 50.24421614])
            self.mean = mnist_mean
            self.std = mnist_std
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize(self.mean, self.std)])
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((111.60893668/255, 113.16127466/255, 120.56512767/255), (50.49768174/255, 51.2589843/255, 50.24421614/255))])



            self.train_set = torchvision.datasets.SVHN(root='./data', split="train",
                                                               download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = torchvision.datasets.SVHN(root='./data', split="test",
                                                              download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "NOTMNIST":
            # MNIST:
            #   type               : torch.ByteTensor
            #   shape              : train_set.train_data.shape torch.Size([60000, 28, 28])
            #   test data shape    : [10000, 28, 28]
            #   number of channels : 1
            #   Mean per channel   : nm.train_data.type(torch.FloatTensor).mean() is 106.51712372448979
            #   Std per channel    : nm.train_data.type(torch.FloatTensor).std() is 115.76734631096612
            self.mean = mnist_mean
            self.std = mnist_std
            transform = transforms.Compose(
                [transforms.Pad(2),
                 transforms.ToTensor(),
                 transforms.Normalize((106.51712372448979 / 255,), (115.76734631096612 / 255,))])

            self.train_set = NOTMNIST(root='./data/notmnist', train=True, download=True, transform=transform)

            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=pin_memory)

            self.test_set = NOTMNIST(root='./data/notmnist', train=False, download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=self.num_workers,
                                                           pin_memory=pin_memory)
        if dataset == "CONTPERMUTEDPADDEDMNIST" or dataset == "CONTPERMUTEDMNIST":

            # if dataset == "CONTPERMUTEDPADDEDMNIST":
            #     transform = transforms.Compose(
            #         [transforms.Pad(2, fill=0, padding_mode='constant'),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=(0.1000,), std=(0.2752,))])

            if dataset == "CONTPERMUTEDPADDEDMNIST":
                transform = transforms.Compose(
                    [transforms.Pad(2, fill=0, padding_mode='constant'),
                    transforms.ToTensor()])
            
            if dataset == "CONTPERMUTEDMNIST":
                transform = transforms.Compose(
                    [transforms.ToTensor()])

            # Original MNIST
            tasks_datasets = [torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)]
            total_len = len(tasks_datasets[0])
            test_loaders = [torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False,
                                                                                   download=True, transform=transform),
                                                        batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=pin_memory)]
            self.num_of_permutations = len(kwargs.get("all_permutation"))
            all_permutation = kwargs.get("all_permutation", None)
            tasks_samples_indices = [torch.tensor(range(len(tasks_datasets[0])), dtype=torch.int32)]
            for p_idx in range(self.num_of_permutations):
                # Create permuation
                permutation = all_permutation[p_idx]

                # Add train set:
                tasks_datasets.append(Permutation(torchvision.datasets.MNIST(root='./data', train=True,
                                                                             download=True, transform=transform),
                                                  permutation, target_offset=0))

                if not self.federated_learning:
                    tasks_samples_indices.append(torch.tensor(range(total_len,
                                                                total_len + len(tasks_datasets[-1])
                                                                ), dtype=torch.int32))
                total_len += len(tasks_datasets[-1])
                # Add test set:
                test_set = Permutation(torchvision.datasets.MNIST(root='./data', train=False,
                                                                  download=True, transform=transform),
                                       permutation, self.target_offset)
                test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                shuffle=False, num_workers=self.num_workers,
                                                                pin_memory=pin_memory))
            self.test_loader = test_loaders
            # Concat datasets
            total_iters = kwargs.get("total_iters", None)

            assert total_iters is not None
            beta = kwargs.get("contpermuted_beta", 3)

            all_datasets = torch.utils.data.ConcatDataset(tasks_datasets)

            tasks_probs_over_iterations_lst = _create_probabilites_over_iterations(total_iters,self.num_of_permutations + 1,beta)
            
            self.tasks_probs_over_iterations = tasks_probs_over_iterations_lst

            # print(len(self.tasks_probs_over_iterations),self.tasks_probs_over_iterations[5800])

            # Create probabilities of tasks over iterations

            #We need to generate a client specific tasks_samples_indices object

            

            if self.federated_learning:
                task_end_iters = [_create_task_probs(total_iters,self.num_of_permutations+1,task_id,beta)[1]
                                    for task_id in range(self.num_of_permutations+1)]

                # round_end_iters = np.linspace(0, total_iters, self.n_fl_rounds + 1, endpoint=True, dtype=int)
                # round_end_iters = round_end_iters[1:]

                task_end_iters.insert(0, 0)
                round_end_iters = np.concatenate([np.linspace(start, stop, self.num_aggs_per_task + 1, endpoint=True)[:-1] for start, stop in zip(task_end_iters[:-1], task_end_iters[1:])])
                round_end_iters = np.append(round_end_iters, task_end_iters[-1])
                round_end_iters = round_end_iters[1:].astype(int)


                print(round_end_iters)
                print(len(round_end_iters))
                
            
                clients_tasks_samples_indices = generate_client_datasets(tasks_datasets, self.non_iid_split, self.n_clients)
                self.client_train_loaders = []
                for client_id in range(self.n_clients):
                    client_train_sampler = FederatedContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                            tasks_samples_indices=clients_tasks_samples_indices[client_id],
                                                            tasks_probs_over_iterations=self.tasks_probs_over_iterations,
                                                            round_end_iter_lst=round_end_iters) 
                    
                    train_loader = torch.utils.data.DataLoader(all_datasets,batch_size=self.batch_size,
                                                               num_workers=self.num_workers,sampler=client_train_sampler,pin_memory=pin_memory)

                    self.client_train_loaders.append(train_loader) 
            else:
                train_sampler = ContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                            tasks_samples_indices=tasks_samples_indices,
                                                            tasks_probs_over_iterations=
                                                                self.tasks_probs_over_iterations,
                                                            num_of_batches=kwargs.get("iterations_per_virtual_epc", 1))
                
                self.train_loader = torch.utils.data.DataLoader(all_datasets, batch_size=self.batch_size,
                                                            num_workers=self.num_workers, sampler=train_sampler, pin_memory=pin_memory)


        if dataset == "CONTSPLITMNIST":

            transform = transforms.Compose(
                    [transforms.ToTensor()])

            classes_lst = [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]
            ]


            full_train_mnist_dataset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)

            full_test_mnist_dataset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

            tasks_datasets = []
            test_datasets = []

            # print(f"full_train_mnist_dataset id {id(full_train_mnist_dataset)}")

            for cl_list in classes_lst:
                mnist_sub_dataset = torch.utils.data.Subset(copy.deepcopy(full_train_mnist_dataset),indices = get_classes_idx_list(copy.deepcopy(full_train_mnist_dataset),cl_list))
                
                # print(f"mnist_sub_dataset id {id(mnist_sub_dataset.dataset)}")
                mnist_sub_dataset = relabel_classes(mnist_sub_dataset)

                mnist_test_sub_dataset = torch.utils.data.Subset(copy.deepcopy(full_test_mnist_dataset),indices = get_classes_idx_list(copy.deepcopy(full_test_mnist_dataset),cl_list))
                
                mnist_test_sub_dataset = relabel_classes(mnist_test_sub_dataset)

                tasks_datasets.append(mnist_sub_dataset)
                test_datasets.append(mnist_test_sub_dataset)
    
        
            total_len = len(tasks_datasets[0])

            test_loaders = []
            for test_dataset in test_datasets:
                # test_loaders = [torch.utils.data.DataLoader(test_dataset,batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, pin_memory=pin_memory)]
                test_loaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory))

            tasks_samples_indices = [torch.tensor(range(len(tasks_datasets[0])), dtype=torch.int32)]
            for _ in range(len(classes_lst)):
                if not self.federated_learning:
                    tasks_samples_indices.append(torch.tensor(range(total_len,
                                                                total_len + len(tasks_datasets[-1])
                                                                ), dtype=torch.int32))
                total_len += len(tasks_datasets[-1])
                
            
            self.test_loader = test_loaders
            # Concat datasets
            total_iters = kwargs.get("total_iters", None)

            assert total_iters is not None
            beta = kwargs.get("contpermuted_beta", 3)

            all_datasets = torch.utils.data.ConcatDataset(tasks_datasets)

            tasks_probs_over_iterations_lst = _create_probabilites_over_iterations(total_iters, len(classes_lst), beta)
            
            self.tasks_probs_over_iterations = tasks_probs_over_iterations_lst

            print(len(self.tasks_probs_over_iterations),self.tasks_probs_over_iterations[5800])

            # Create probabilities of tasks over iterations

            #We need to generate a client specific tasks_samples_indices object

            if self.federated_learning:
                round_end_iters = [_create_task_probs(total_iters,len(classes_lst),task_id,beta)[1]
                                    for task_id in range(len(classes_lst))]

                print(round_end_iters)

                print("*"*100)
                print(f"non_iid {self.non_iid_split}")
                print("*"*100)

                clients_tasks_samples_indices = generate_client_datasets(tasks_datasets, self.non_iid_split, self.n_clients)
                self.client_train_loaders = []
                for client_id in range(self.n_clients):
                    client_train_sampler = FederatedContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                            tasks_samples_indices=clients_tasks_samples_indices[client_id],
                                                            tasks_probs_over_iterations=self.tasks_probs_over_iterations,
                                                            round_end_iter_lst=round_end_iters)
                    
                    train_loader = torch.utils.data.DataLoader(all_datasets,batch_size=self.batch_size,
                                                               num_workers=self.num_workers,sampler=client_train_sampler,pin_memory=pin_memory)

                    self.client_train_loaders.append(train_loader)
            else:
                train_sampler = ContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                            tasks_samples_indices=tasks_samples_indices,
                                                            tasks_probs_over_iterations=
                                                                self.tasks_probs_over_iterations,
                                                            num_of_batches=kwargs.get("iterations_per_virtual_epc", 1))
                
                self.train_loader = torch.utils.data.DataLoader(all_datasets, batch_size=self.batch_size,
                                                            num_workers=self.num_workers, sampler=train_sampler, pin_memory=pin_memory)        

         
class ContinuousMultinomialSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    self.tasks_probs_over_iterations is the probabilities of tasks over iterations.
    self.samples_distribution_over_time is the actual distribution of samples over iterations
                                            (the result of sampling from self.tasks_probs_over_iterations).
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, samples_in_batch=128, num_of_batches=69, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):
        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.num_of_batches = num_of_batches

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))
    
    def generate_iters_indices(self, num_of_iters):
        from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, from_iter+num_of_iters):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return len(self.samples_in_batch)


class FederatedContinuousMultinomialSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    self.tasks_probs_over_iterations is the probabilities of tasks over iterations.
    self.samples_distribution_over_time is the actual distribution of samples over iterations
                                            (the result of sampling from self.tasks_probs_over_iterations).
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source,round_end_iter_lst, samples_in_batch=128, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):
        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.round_end_iter_lst = round_end_iter_lst

        self.from_iter = 0
        self.to_iter_idx = 0

        self.current_round_start_iter = 0
        self.current_round_end_iter = self.round_end_iter_lst[self.to_iter_idx]

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))
    
    def generate_iters_indices(self, from_iter, to_iter):
        # from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, to_iter):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        self.generate_iters_indices(self.from_iter,self.round_end_iter_lst[self.to_iter_idx])

        self.current_round_start_iter = self.from_iter
        self.current_round_end_iter = self.round_end_iter_lst[self.to_iter_idx]

        self.from_iter = self.round_end_iter_lst[self.to_iter_idx]
        self.to_iter_idx+=1

        self.current_iteration = self.from_iter

        return iter([item for sublist in 
                     self.iter_indices_per_iteration[self.current_round_start_iter:self.current_round_end_iter] for item in sublist])

    def __len__(self):
        return (self.current_round_end_iter - self.current_round_start_iter)*(self.samples_in_batch)


def _get_linear_line(start, end, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i - start)/(end-start) for i in range(start, end)])
    return torch.FloatTensor([1 - ((i - start) / (end - start)) for i in range(start, end)])


def _create_task_probs(iters, tasks, task_id, beta=3):
    if beta <= 1:
        peak_start = int((task_id/tasks)*iters)
        peak_end = int(((task_id + 1) / tasks)*iters)
        start = peak_start
        end = peak_end
    else:
        start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
        peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
        peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
        end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

    #This is a probability dist for each task across iterations
    probs = torch.zeros(iters, dtype=torch.float)

    if task_id == 0:
        probs[start:peak_start].add_(1)
    else:
        probs[start:peak_start] = _get_linear_line(start, peak_start, direction="up")
    probs[peak_start:peak_end].add_(1)
    if task_id == tasks - 1:
        probs[peak_end:end].add_(1)
    else:
        probs[peak_end:end] = _get_linear_line(peak_end, end, direction="down")
    
    # with open('probs.txt','w') as f:
    #     f.write(str(probs.numpy().tolist()))

    return probs,end


###
# NotMNIST
###
class NOTMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-images-idx3-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/t10k-labels-idx1-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-images-idx3-ubyte.gz',
        'https://github.com/davidflanagan/notMNIST-to-MNIST/raw/master/train-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            self.read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            self.read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            self.read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            self.read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    def read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2049
            length = self.get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    def read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2051
            length = self.get_int(data[4:8])
            num_rows = self.get_int(data[8:12])
            num_cols = self.get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)


###########################################################################
# Callable datasets
###########################################################################
def ds_split_mnist(**kwargs):
    """
    Split MNIST dataset. Consists of 5 tasks: digits 0 & 1, 2 & 3, 4 & 5, 6 & 7, and 8 & 9.
    :param batch_size: batch size
           num_workers: num of workers
           pad_to_32: If true, will pad digits to size 32x32 and normalize to zero mean and unit variance.
           separate_labels_space: If true, each task will have its own label space (e.g. 01, 23 etc.).
                                  If false, all tasks will have label space of 0,1 only.
    :return: Tuple with two lists.
             First list of the tuple is a list of 5 train loaders, each loader is a task.
             Second list of the tuple is a list of 5 test loaders, each loader is a task.
    """
    classes_lst = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ]
    classes_lst = kwargs.get("classes_lst", classes_lst)
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               reduce_classes=cl, pad_to_32=kwargs.get("pad_to_32", False),
                               preserve_label_space=kwargs.get("separate_labels_space")) for cl in classes_lst]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_padded_split_mnist(**kwargs):
    """
    Split MNIST dataset, padded to 32x32 pixels.
    """
    return ds_split_mnist(pad_to_32=True, **kwargs)


def ds_split_mnist_offline(**kwargs):
    """
    Split MNIST dataset. Offline means that all tasks are mixed together.
    """
    if kwargs.get("separate_labels_space"):
        return ds_mnist(**kwargs)
    else:
        return ds_mnist(labels_remapping={l: l % 2 for l in range(10)}, **kwargs)


def ds_padded_split_mnist_offline(**kwargs):
    """
    Split MNIST dataset. Padded to 32x32. Offline means that all tasks are mixed together.
    """
    return ds_split_mnist_offline(pad_to_32=True, **kwargs)


def ds_permuted_mnist(**kwargs):
    """
    Permuted MNIST dataset.
    First task is the MNIST datasets (with 10 possible labels).
    Other tasks are permutations (pixel-wise) of the MNIST datasets (with 10 possible labels).
    :param batch_size: batch size
           num_workers: num of workers
           pad_to_32: If true, will pad digits to size 32x32 and normalize to zero mean and unit variance.
           permutations: A list of permutations. Each permutation should be a list containing new pixel position.
           separate_labels_space: True for seperated labels space - task i labels will be (10*i) to (10*i + 9).
                                  False for unified labels space - all tasks will have labels of 0 to 9.
    :return: Tuple with two lists.
             First list of the tuple is a list of train loaders, each loader is a task.
             Second list of the tuple is a list of test loaders, each loader is a task.
    """
    # First task
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), pad_to_32=kwargs.get("pad_to_32", False))]
    target_offset = 0
    permutations = kwargs.get("permutations", [])
    for pidx in range(len(permutations)):
        if kwargs.get("separate_labels_space"):
            target_offset = (pidx + 1) * 10
        dataset.append(DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                                       num_workers=kwargs.get("num_workers", 1),
                                       permutation=permutations[pidx], target_offset=target_offset,
                                       pad_to_32=kwargs.get("pad_to_32", False)))
    # For offline permuted we take the datasets and mix them.
    if kwargs.get("offline", False):
        train_sets = []
        test_sets = []
        for ds in dataset:
            train_sets.append(ds.train_set)
            test_sets.append(ds.test_set)
        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=kwargs.get("batch_size", 128), shuffle=True,
                                                   num_workers=kwargs.get("num_workers", 1), pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=kwargs.get("batch_size", 128), shuffle=False,
                                                  num_workers=kwargs.get("num_workers", 1), pin_memory=True)
        return [train_loader], [test_loader]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_padded_permuted_mnist(**kwargs):
    """
    Permuted MNIST dataset, padded to 32x32.
    """
    return ds_permuted_mnist(pad_to_32=True, **kwargs)


def ds_permuted_mnist_offline(**kwargs):
    """
    Permuted MNIST dataset. Offline means that all tasks are mixed together.
    """
    return ds_permuted_mnist(offline=True, **kwargs)


def ds_padded_permuted_mnist_offline(**kwargs):
    """
    Permuted MNIST dataset, padded to 32x32. Offline means that all tasks are mixed together.
    """
    return ds_permuted_mnist(pad_to_32=True, offline=True, **kwargs)



'''This is an important method which actually sends in all keyword arguments and other stuff that provides 
us an insight into the parameters that DatasetsLoaders class takes in and operates with.
'''
def ds_padded_cont_permuted_mnist(**kwargs):
    """
    Continuous Permuted MNIST dataset, padded to 32x32.
    Notice that this dataloader is aware to the epoch number, therefore if the training is loaded from a checkpoint
        adjustments might be needed. 
    Access dataset.tasks_probs_over_iterations to see the tasks probabilities for each iteration.
    :param num_epochs: Number of epochs for the training (since it builds distribution over iterations,
                            it needs this information in advance)
    :param iterations_per_virtual_epc: In continuous task-agnostic learning, the notion of epoch does not exists,
                                        since we cannot define 'passing over the whole dataset'. Therefore,
                                        we define "iterations_per_virtual_epc" -
                                        how many iterations consist a single epoch.
    :param contpermuted_beta: The proportion in which the tasks overlap. 4 means that 1/4 of a task duration will
                                consist of data from previous/next task. Larger values means less overlapping.
    :param permutations: The permutations which will be used (first task is always the original MNIST).
    :param batch_size: Batch size.
    :param num_workers: Num workers.
    :return: A tuple of (train_loaders, test_loaders). train_loaders is a list of 1 data loader - it loads the
                permuted MNIST dataset continuously as described in the paper. test_loaders is a list of 1+permutations
                data loaders, one for each dataset.

    """
    dataset = [DatasetsLoaders("CONTPERMUTEDPADDEDMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               total_iters=(kwargs.get("num_epochs")*kwargs.get("iterations_per_virtual_epc")),
                               contpermuted_beta=kwargs.get("contpermuted_beta"),
                               iterations_per_virtual_epc=kwargs.get("iterations_per_virtual_epc"),
                               all_permutation=kwargs.get("permutations", []))]
    test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
    train_loaders = [ds.train_loader for ds in dataset]

    return train_loaders, test_loaders


#Customizing below method to include federated learning setting as we
#Same as the above method,only thing being padded has been removed and image size would be 28x28 only.
def ds_cont_permuted_mnist(**kwargs):
    """
    Continuous Permuted MNIST dataset
    Notice that this dataloader is aware to the epoch number, therefore if the training is loaded from a checkpoint
        adjustments might be needed. 
    Access dataset.tasks_probs_over_iterations to see the tasks probabilities for each iteration.
    :param num_epochs: Number of epochs for the training (since it builds distribution over iterations,
                            it needs this information in advance)
    :param iterations_per_virtual_epc: In continuous task-agnostic learning, the notion of epoch does not exists,
                                        since we cannot define 'passing over the whole dataset'. Therefore,
                                        we define "iterations_per_virtual_epc" -
                                        how many iterations consist a single epoch.
    :param contpermuted_beta: The proportion in which the tasks overlap. 4 means that 1/4 of a task duration will
                                consist of data from previous/next task. Larger values means less overlapping.
    :param permutations: The permutations which will be used (first task is always the original MNIST).
    :param batch_size: Batch size.
    :param num_workers: Num workers.
    :return: A tuple of (train_loaders, test_loaders). train_loaders is a list of 1 data loader - it loads the
                permuted MNIST dataset continuously as described in the paper. test_loaders is a list of 1+permutations
                data loaders, one for each dataset.

    """
    dataset = [DatasetsLoaders("CONTPERMUTEDMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               total_iters=(kwargs.get("num_epochs")*kwargs.get("iterations_per_virtual_epc")),
                               contpermuted_beta=kwargs.get("contpermuted_beta"),
                               iterations_per_virtual_epc=kwargs.get("iterations_per_virtual_epc"),
                               all_permutation=kwargs.get("permutations", []),fl=kwargs.get("federated_learning",False),n_clients=kwargs.get("n_clients",5), num_aggs_per_task = kwargs.get("num_aggs_per_task", 5), non_iid_split = kwargs.get("non_iid_split", False))]
    
    test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
    
    if kwargs.get("federated_learning",False):
        train_loaders = dataset[0].client_train_loaders
    else:
        train_loaders = [ds.train_loader for ds in dataset]

    return train_loaders, test_loaders


def ds_cont_split_mnist(**kwargs):

    #Add code for a flexible classes_list - default classes list - [[0,1],[2,3],[4,5],[6,7],[8,9]]

    dataset = [DatasetsLoaders("CONTSPLITMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               total_iters=(kwargs.get("num_epochs")*kwargs.get("iterations_per_virtual_epc")),
                               contpermuted_beta=kwargs.get("contpermuted_beta"),
                               iterations_per_virtual_epc=kwargs.get("iterations_per_virtual_epc"),
                               fl=kwargs.get("federated_learning",False),n_clients=kwargs.get("n_clients",5), non_iid_split = kwargs.get("non_iid_split", False))]
    
    test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
    
    if kwargs.get("federated_learning",False):
        train_loaders = dataset[0].client_train_loaders
    else:
        train_loaders = [ds.train_loader for ds in dataset]

    return train_loaders, test_loaders


def ds_visionmix(**kwargs):
    """
    Vision mix dataset. Consists of: MNIST, notMNIST, FashionMNIST, SVHN and CIFAR10.
    """
    dataset = [DatasetsLoaders("MNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), pad_to_32=True),
               DatasetsLoaders("NOTMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("FashionMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("SVHN", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1)),
               DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1))]
    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_cifar10and100(**kwargs):
    """
    CIFAR10 and CIFAR100 dataset. Consists of 6 tasks:
        1) CIFAR10
        2-6) Subsets of 10 classes from CIFAR100.
    """
    classes_lst = [[j for j in range(i * 10, (i + 1) * 10)] for i in range(0, 5)]
    dataset = [DatasetsLoaders("CIFAR100", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               reduce_classes=cl, preserve_label_space=False) for cl in classes_lst]
    dataset = [DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1), preserve_label_space=False)] + dataset

    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders


def ds_cifar10(**kwargs):
    """
    CIFAR10 dataset. No tasks.
    """
    dataset = [DatasetsLoaders("CIFAR10", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1))]

    test_loaders = [ds.test_loader for ds in dataset]
    train_loaders = [ds.train_loader for ds in dataset]
    return train_loaders, test_loaders
