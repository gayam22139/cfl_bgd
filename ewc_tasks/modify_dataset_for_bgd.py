import torch
import pickle

datasets = pickle.load(open('ewc_tasks/permuted_mnist_tasks(3).pkl','rb'))

def get_data_train_test_data(datasets,dataset_idx):
    mnist_dataset = datasets[dataset_idx]
    train_dataset,test_dataset = mnist_dataset[0],mnist_dataset[1]
    x_train,y_train = torch.tensor(train_dataset[0]),torch.tensor(train_dataset[1])
    x_test,y_test = torch.tensor(test_dataset[0]),torch.tensor(test_dataset[1])

    return x_train,y_train,x_test,y_test


def modify_labels(y_train,y_test,dataset_idx):
    
    modified_y_train = y_train.apply_(lambda label:10*dataset_idx+label)
    modified_y_test = y_test.apply_(lambda label:10*dataset_idx+label)

    return modified_y_train,modified_y_test


mnist_x_train,mnist_y_train,mnist_x_test,mnist_y_test = get_data_train_test_data(datasets,0)
perm_mnist1_x_train,perm_mnist1_y_train,perm_mnist1_x_test,perm_mnist1_y_test = get_data_train_test_data(datasets,1)
perm_mnist2_x_train,perm_mnist2_y_train,perm_mnist2_x_test,perm_mnist2_y_test = get_data_train_test_data(datasets,2)

modified_perm_mnist1_y_train,modified_perm_mnist1_y_test = modify_labels(perm_mnist1_y_train,perm_mnist1_y_test,1)

modified_perm_mnist2_y_train,modified_perm_mnist2_y_test  = modify_labels(perm_mnist2_y_train,perm_mnist2_y_test,2)

def create_dataset(x_train,y_train,x_test,y_test,idx,datasets = {}):
    train_data = (x_train,y_train)
    test_data = (x_test,y_test)

    datasets[idx] = (train_data,test_data)

    return datasets

datasets = create_dataset(mnist_x_train,mnist_y_train,mnist_x_test,mnist_y_test,0)
datasets = create_dataset(perm_mnist1_x_train,perm_mnist1_y_train,perm_mnist1_x_test,perm_mnist1_y_test,1,datasets)
datasets = create_dataset(perm_mnist2_x_train,perm_mnist2_y_train,perm_mnist2_x_test,perm_mnist2_y_test,2,datasets)

# print(len(datasets))
# print(datasets[2][0][1])

#save the modified_datasets_dict_to_a_pickle_file
pickle.dump(datasets,open('permuted_mnist_tasks_modified_labels(3).pkl','wb'))