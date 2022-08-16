import torch
import os
from collections import Counter
import numpy as np
import pickle
import copy
from torch.utils.data import Dataset
from PIL import Image
import tarfile
import shutil
from typing import Any, Callable, Optional, Tuple
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST


def partition_data(dataset, datadir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, download=False)
        test_ds = dl_obj(datadir, train=False, download=False)

        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=False)
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom

        train_ds = dl_obj(datadir+'./train/', dataidxs=dataidxs)
        test_ds = dl_obj(datadir+'./val/')

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    return train_dl, test_dl, train_ds, test_ds



class CIFAR10_truncated(torch.utils.data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
################# the alpha controlls the partition ######################
alpha = 0.5
all_clients = 10

X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
"cifar10", "../data", "noniid", all_clients, beta=alpha)

copy_sta = copy.deepcopy(traindata_cls_counts)

with open("traindata_cls_counts.pkl","wb") as file:
    pickle.dump(traindata_cls_counts,file)

## put the data that after partition into disk from memory, so CAREFULLY DO IT TO NOT OVERWRITE THE EXISTING DATA
## WARNING: CHECK HTE PATH NO DATA
keep_avg_align = False
for client in range(10):
    
    ## ds mean all the data, only dl is related to batchsize
    train_dl_local, test_dl_local, train_ds, test_ds = get_dataloader("cifar10", 
                                                                      '../data', 64, 32,
                                                                      dataidxs=net_dataidx_map[client])


    transform_type = torchvision.transforms.Compose([
        transforms.ToTensor()
    ])
    
    data_ = [transform_type(i) for i in train_ds.data]
    data_ = np.vstack(data_).reshape(-1, 3, 32, 32).transpose((0,2,3,1))
    labels_=list(train_ds.target)
    origin_lab = np.array(labels_)
    
    transform_aug = torchvision.transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(32),
        torchvision.transforms.ToTensor()
    ])
    # i stands for labels
    for i in range(10):
        data_index = np.argwhere(origin_lab==i).reshape(-1)
        if i not in Counter(train_ds.target).keys():
            # add 20 noise to those who lossing classes
            for j in range(20):
                t_d = data_[0].shape
                noise = torch.randn(t_d).reshape((1,32,32,3)).numpy()
                data_=np.concatenate((data_, noise))
                labels_.append(i)
                copy_sta[client][i] = 1 if j==0 else copy_sta[client][i]+1
        elif traindata_cls_counts[client][i]<20:
            # add 10 aug images to those who has less than 10 images' classes, switch to dataset using transform
            num_da = traindata_cls_counts[client][i]
            while copy_sta[client][i]<20:
                for j in data_index:
                    aug = transform_aug(data_[j]).cpu().numpy().transpose((1,2,0)).reshape((1,32,32,3))
                    
                    data_=np.concatenate((data_, aug))
                    labels_.append(i)
                    copy_sta[client][i]+=1
        ### keep align with the avg num class
        if keep_avg_align:
            data_index_new = np.argwhere(np.array(labels_)==i).reshape(-1)
            mean_num_client = int(np.array(list(traindata_cls_counts[client].values())).mean())
            while copy_sta[client][i] < mean_num_client:
                choice_index = np.random.choice(data_index_new)
                aug = transform_aug(data_[choice_index]).cpu().numpy().transpose((1,2,0)).reshape((1,32,32,3))
                data_=np.concatenate((data_, aug))
                labels_.append(i)
                copy_sta[client][i]+=1
                
            
    num = data_.shape[0]
    step = num//5
    data_ = data_.reshape(num,-1)
    os.makedirs("cifar-10-batches-py",exist_ok=True)
    for i in range(5):
        dic = {}
        dic['data']=data_[step*i:step*(i+1)]
        dic['labels']=labels_[step*i:step*(i+1)]
        with open("cifar-10-batches-py/data_batch_"+str(i+1),"wb") as f:
            pickle.dump(dic,f)

    # compress file to tar.gz
    def Converter(path, tar):
        with tarfile.open(tar, "w:gz") as t:
            for root, dirs, files in os.walk(path):
                for file in files:
                    t.add(os.path.join(root, file))

    Converter("cifar-10-batches-py", "cifar-10-python.tar.gz")
    # move to new dir
    ne_path = "data/cifar10/alpha-"+str(alpha)+"/partition_client_"+str(client+1)
    os.makedirs(ne_path) #,exist_ok=True) in order to not override the origional data
    shutil.move("cifar-10-python.tar.gz",ne_path)


## uncompress
import subprocess
current_dir = os.getcwd()
print(current_dir)
for client in range(10):
    xpath = "data/cifar10/alpha-"+str(alpha)+"/partition_client_"+str(client+1)
    os.chdir(xpath)
    subprocess.run(["tar","-xzvf","cifar-10-python.tar.gz"])
    os.chdir(current_dir)


## related to the partition alpha 
import matplotlib.pyplot as plt
for flag in ["former","augmentation"]:
    if flag == "former":
        x = traindata_cls_counts
    else:
        x = copy_sta
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row',figsize=(25,10))
    for i in range(10):
        if i==0:
            ax[i//5,i%5].set_ylabel("Number of corresponding label")
        elif i>5:
            ax[i//5,i%5].set_xlabel("Class_label")
        elif i==5:
            ax[i//5,i%5].set_ylabel("Number of corresponding label")
            ax[i//5,i%5].set_xlabel("Class_label")

        ax[i//5,i%5].set_xlim(-1,10)
        ax[i//5,i%5].bar(list(x[i].keys()),list(x[i].values()),width = 1)
        ax[i//5,i%5].set_title("client-{0}".format(i))
        for m,n in zip(list(x[i].keys()),list(x[i].values())):
            ax[i//5,i%5].text(m+0.05,n+0.05,'%d' %n, ha='center',va='bottom')
    # plt.title("partition-"+str(alpha)) not work
    plt.savefig("data/cifar10/alpha-"+str(alpha)+"/"+flag+"_partition.png",dpi=330)


