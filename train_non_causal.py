import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
# from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import ZINC

import warnings

warnings.filterwarnings('ignore')
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import Evaluator
from torch import tensor
from train_epoch import train_epoch
import numpy as np
import os
from opts_GOOD import print_args, parse_args
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_zinc import GOODZINC
from GOOD.data.good_datasets.good_cyc import GOODCYC
# from GOOD.data.good_datasets.orig_zinc import ZINC
from model_GOOD import GCNNet, GINNet, GATNet
import time
import random
import pdb

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def main(args, trail):

    set_seed(args.seed)
        
    if args.dataset == "zinc":
        dataset, meta_info = GOODZINC.load(args.data_root,
                                           domain=args.domain,
                                           shift=args.shift,
                                           generate=False)
        args.num_classes = 1  
        args.eval_metric = "mae"  
        args.dim_node = meta_info['dim_node']
        args.layers = 3
        # args.eval_name = "ogbg-molzinc"
    
    elif args.dataset == "cyc":
        dataset, meta_info = GOODCYC.load(args.data_root,
                                           domain=args.domain,
                                           shift=args.shift,
                                           generate=False)
        args.num_classes = 1  
        args.eval_metric = "mae"  
        args.dim_node = meta_info['dim_node']
        args.layers = 3
        # args.eval_name = "ogbg-molzinc"

    elif args.dataset == "ori_zinc": 
        # dataset = TUDataset(root=args.data_root, name="ZINC_full", use_node_attr=True, use_edge_attr=True)
        # print("TUdataset:", dataset[0])
        train_dataset = ZINC(root = args.data_root, split='train') # valid, test
        val_dataset = ZINC(root = args.data_root, split='val')
        test_dataset = ZINC(root = args.data_root, split='test')
        args.num_classes = 1  
        args.eval_metric = "mae"  
        args.dim_node = train_dataset.num_features
        args.layers = 3
        dataset = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
        }
    elif args.dataset == "hiv":
        dataset, meta_info = GOODHIV.load(args.data_root,
                                          domain=args.domain,
                                          shift=args.shift,
                                          generate=False)
        args.num_classes = 2
        args.dim_node = meta_info['dim_node']
        # args.layers = 3
        args.eval_metric = "rocauc"
        args.eval_name = "ogbg-molhiv"
        # args.task_type = dataset['task']


    else:
        assert False

    random_guess = 1.0 / args.num_classes
    evaluator = Evaluator(args.eval_name)
    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader_ood = DataLoader(
        dataset["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader_ood = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GCNNet(args.dim_node, args.num_classes, args).to(device)

    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=args.epochs,
                                     eta_min=args.min_lr,
                                     last_epoch=-1,
                                     verbose=False)

    results = {'highest_valid': 100, 'update_test': 100, 'update_epoch': 100}
    start_time = time.time()
    prototype = None
    memory_bank = 0.5 * torch.ones(args.batch_size * args.me_batch_n,
                                   args.hidden).to(device)
    for epoch in range(1, args.epochs + 1):
        start_time_local = time.time()
        train_loss = train_epoch(
            model, optimizer, train_loader, device, memory_bank, prototype,
            criterion, args)

        valid_result = eval_non_causal(model, valid_loader_ood, device, args, evaluator)
        test_result = eval_non_causal(model, test_loader_ood, device, args, evaluator)
        lr_scheduler.step()
        if valid_result < results['highest_valid'] and epoch > 10:
            results['highest_valid'] = valid_result
            results['update_test'] = test_result
            results['update_epoch'] = epoch

        print("-" * 150)
        print(
            "Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}] Valid:[{:.4f}] Test:[{:.4f}]  (RG:{:.4f}) | Best Valid:[{:.4f}] Update test:[{:.4f}] at Epoch:[{}] | epoch time[{:.2f}min "
            .format(args.dataset, trail, epoch, args.epochs, train_loss, valid_result, test_result,
                    random_guess, results['highest_valid'],
                    results['update_test'], results['update_epoch'],
                    (time.time() - start_time_local) / 60))

        print("-" * 150)
    total_time = time.time() - start_time

    print(
        "mwy: Causal fold:[{}] | Dataset:[{}] | Update Test:[{:.4f}] at epoch [{}] | (RG:{:.4f}) | Total time:{}"
        .format(trail, args.dataset, results['update_test'],
                results['update_epoch'], random_guess,
                time.strftime('%H:%M:%S', time.gmtime(total_time))))

    print("-" * 150)
    print('\n')
    # final_test_iid.append(results['update_test_iid'])
    return results['update_test']


def config_and_run(args):

    print_args(args)
    # set_seed(args.seed)
    final_test = []
    for trail in range(args.trails):
        test_result = main(args, trail + 1)
        final_test.append(test_result)
    print("mwy finall: Test result: [{:.2f}±{:.2f}]".format(
        np.mean(final_test) * 100,
        np.std(final_test) * 100))
    print("ALL OOD:{}".format(final_test))



def eval_non_causal(model, loader, device, args, evaluator):
    
    model.eval()
    if args.eval_metric == "mae": 
        y_true, y_pred = [], []

        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    y_logs = model(data)

                    y_true.append(data.y.view(y_logs.shape).detach().cpu())
                    y_pred.append(y_logs.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        mae = torch.mean(torch.abs(torch.tensor(y_true) - torch.tensor(y_pred)))
        output_o = mae

    else:
        assert False
    return output_o



if __name__ == "__main__":
    args = parse_args()
    config_and_run(args)
    print(
        "settings | beta:[{}]  n:[{}]  prototype/memory:[{}/{}]  dim_node:[{}] num_classes:[{}] batch_size:[{}]  hidden:[{}] lr:[{}] min_lr:[{}] weight_decay[{}] "
        .format(str(args.beta), str(args.me_batch_n), str(args.prototype),
                str(args.memory), str(args.dim_node), str(args.num_classes),
                str(args.batch_size), str(args.hidden), str(args.lr),
                str(args.min_lr), str(args.weight_decay)))

    print("-" * 150)
