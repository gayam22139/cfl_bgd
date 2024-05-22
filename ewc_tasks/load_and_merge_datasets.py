import pickle
import numpy as np
import random

seed1 = 0
seed2 = 1

loaded_datasets = pickle.load(open('permuted_mnist_tasks_modified_labels(3).pkl','rb'))

d1=loaded_datasets[0]
d2=loaded_datasets[1]

print("length of d1 is",len(d1[0][0]))
print("length of d2 is",len(d2[0][0]))

d1_idxs = list(range(len(d1[0][0])))
d2_idxs = list(range(len(d2[0][0])))

def sampling_records(available_idxs1,available_idxs2,batch_size,fraction_1 = 1):
    random_idxs1 = None
    random_idxs2 = None
    remaining_idxs1 = None
    remaining_idxs2 = None

    sample_size1 = round(batch_size * fraction_1)
    sample_size2 = round(batch_size * (1 - fraction_1))


    if len(available_idxs1) <= sample_size1:
        print("Requested number of d1 samples not available and hence all available samples returned")
        random_idxs1 = available_idxs1
        remaining_idxs1 = []
    
    else:
        random.seed(seed1)
        np.random.seed(seed1)
        random_idxs1 = np.random.choice(available_idxs1,sample_size1,replace=False)
        remaining_idxs1 = list(set(available_idxs1) - set(random_idxs1))
        
    if len(available_idxs2) <= sample_size2:
        print("Requested number of d2 samples not available and hence all available samples returned")
        random_idxs2 = available_idxs2
        remaining_idxs2 = []

    else:
        random.seed(seed2)
        np.random.seed(seed2)
        random_idxs2 = np.random.choice(available_idxs2,sample_size2,replace=False)
        remaining_idxs2 = list(set(available_idxs2) - set(random_idxs2))
    
    return random_idxs1,remaining_idxs1,random_idxs2,remaining_idxs2



# s1,r1,s2,r2 = sampling_records(d1_idxs,d2_idxs,10,0.25)
# print(s1)
# print(s2)