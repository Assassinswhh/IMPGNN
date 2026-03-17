from utils import *
from evaluator import Evaluator
from final_model import *
from sklearn.utils import shuffle
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import math
from tqdm import tqdm
import torch_geometric
import argparse
import numpy as np
import datetime
import pickle
from preprocess_data import get_dataset
import warnings
from sklearn.model_selection import train_test_split
import shutil

import time
import psutil
import os

torch.set_printoptions(threshold=float('inf'))
import nni

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

cls_criterion = torch.nn.BCEWithLogitsLoss()
cls_criterion_1 = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.L1Loss()
mse_reg_criterion = torch.nn.MSELoss()
cosine_criterion = torch.nn.CosineEmbeddingLoss()


def train(model, device, loader, optimizer, task_type):
    model.train()
    loss_curve = list()
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, H_mg, H_org = model(batch)

            optimizer.zero_grad()
            is_labeled = batch.y == batch.y

            if "classification" in task_type:
                # loss = proto_align_loss(pred_g, pred_l)
                loss = 0.7 * cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]) \
                       + 0.3 * mse_reg_criterion(H_mg, H_org)

            elif "mse_regression" in task_type:
                loss = 0.7 * mse_reg_criterion(pred.to(torch.float32)[is_labeled],
                                               batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size())) \
                       + 0.3 * mse_reg_criterion(H_mg, H_org)

            loss.backward()
            optimizer.step()
            loss_curve.append(loss.detach().cpu().item())

    return sum(loss_curve) / len(loss_curve)


@torch.no_grad()
def eval(model, device, loader, evaluator, task_type):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    loss_curve = list()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:

            pred, H_mg, H_org = model(batch)

            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                # print(pred.to(torch.float32)[is_labeled].shape, batch.y.to(torch.float32)[is_labeled].shape)
                loss = 0.7 * cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]) \
                       + 0.3 * mse_reg_criterion(H_mg, H_org)

            elif task_type == 'mse_regression':
                loss = 0.7 * mse_reg_criterion(pred.to(torch.float32)[is_labeled],
                                               batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size())) \
                       + 0.3 * mse_reg_criterion(H_mg, H_org)

            loss_curve.append(loss.detach().cpu().item())
            total_loss += loss.item() * batch.num_graphs
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), total_loss / len(loader.dataset), sum(loss_curve) / len(loss_curve)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--patience', default=100, type=int, help='max early stopping round')
    parser.add_argument('--dataset', type=str, default="qm9", help='BBBP, Bace, ClinTox, Tox21, Esol, '
                                                                       'Freesolv, Lipop, Sider, qm9')
    # ctrl+R直接替换 pool/Freesolv 替换为 pool/Freesolv
    parser.add_argument('--num_path', default=30, type=int, help='16, 12, 45, 82, 12, 12, 18, 17, 30')
    # parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay on optimizer.')
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--eval_metric', type=str,
                        default="mae")  ##BBBP BACE ClinTox SIDER 是ROC-AUC剩下的三个ESOL FreeSolv Lipop 是RMSE（代码里面你写小写就行rmse）
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--task_num', type=int, default=1)  ##1，1，2，（tox21不跑），1，1，1，27
    parser.add_argument('--qm9_task_num', type=int, default=6, help='0-11')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seeds', type=int, default=1)

    args = parser.parse_args()

    ## Reproductible results
    torch.manual_seed(3047)
    np.random.seed(3047)

    # Return to benchmarks directory in case we start script for script directory
    if os.getcwd()[-10:] != 'benchmarks':
        os.chdir(os.getcwd())

    ###### LOGGING AND DIRECTORY ######
    if not os.path.exists(os.path.join(os.getcwd(), 'results', args.dataset)):
        os.makedirs(os.path.join(os.getcwd(), 'results', args.dataset))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs', args.dataset)):
        os.makedirs(os.path.join(os.getcwd(), 'logs', args.dataset))
    if not os.path.exists(os.path.join('models', args.dataset)):
        os.makedirs(os.path.join('models', args.dataset))
    now = "_" + "-".join(str(datetime.datetime.today()).split()).split('.')[0].replace(':', '.')
    weight_decay = "wd_" + str(args.weight_decay) if args.weight_decay > 0 else ""
    program_name = f'_bs_{args.batch_size}' + '_lr_' + str(
        args.lr) + weight_decay + now

    logging.basicConfig(filename=os.path.join('logs', args.dataset, program_name + '.log'), level=logging.INFO,
                        filemode="w")
    log = PrinterLogger(logging.getLogger(__name__))
    print(args)

    model_save_path = os.path.join('models', args.dataset, 'model_best' + program_name + '.pth.tar')
    test_perfs_at_best, test_perfs_at_end, test_losses = [], [], []
    val_perfs_at_best, val_perfs_at_end, val_losses = [], [], []
    train_perfs_at_end, train_losses = [], []
    best_epochs = []

    dataset = get_dataset(args.dataset, args.output_dir)

    if args.dataset in ["BBBP", "Bace", "ClinTox", "Sider", "Tox21", "HIV", "MUTAG",
                        "BBBP1", "Bace1", "ClinTox1", "Tox211", "HIV1", "Sider1"]:
        maximize = True
    else:
        maximize = False

    if args.dataset in ["BBBP", "ClinTox", "Tox21", "Sider", "HIV", "Esol1", "Lipop1", "Freesolv1",
                        "Esol", "Freesolv", "Lipop", "Bace", "BBBP1", "Bace1", "ClinTox1", "Tox211",
                        "HIV1", "Sider1"]:
        dataset.data.y = dataset.data.y.masked_fill(torch.isnan(dataset.data.y), 0)
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=math.floor(dataset.data.y.shape[0] * 0.8),
                                          valid_size=math.floor(dataset.data.y.shape[0] * 0.1), seed=42)
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False)
    elif args.dataset in ["qm9"]:
        dataset.data.y = dataset.data.y.masked_fill(torch.isnan(dataset.data.y), 0)
        dataset.data.y = dataset.data.y[:, args.qm9_task_num].unsqueeze(-1)
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=math.floor(dataset.data.y.shape[0] * 0.8),
                                          valid_size=math.floor(dataset.data.y.shape[0] * 0.1), seed=42)
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False)


    evaluator = Evaluator(args.task_num, args.eval_metric)

    n_seeds = args.seeds
    log.print_and_log("\n" + "-" * 15 + f" TESTING OVER {n_seeds} SEEDS" + "-" * 15 + "\n")

    model = WNet(num_path=args.num_path, hidden_dim=args.hidden_dim,
                 task_num=args.task_num, k=args.K, p=args.dropout)

    fs = []
    log.print_and_log(f'Model # Parameters {sum([p.numel() for p in model.parameters()])}')
    for seed in range(n_seeds):
        model.reset_parameters()
        if args.weight_decay == 0:
            optim_class = optim.Adam
        else:
            optim_class = optim.AdamW

        optimizer = optim_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.dataset in ["Esol", "Freesolv", "Lipop", "Esol1", "Freesolv1", "Lipop1", "qm9"]:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.1, mode="min")
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode="max")

        # 分类的时候最后一个是Ture， 回归是False
        early_stopper = Patience(patience=args.patience, use_loss=False, save_path=model_save_path, maximize=maximize)

        pbar = tqdm(range(args.epochs), desc=f"{seed + 1}/{n_seeds}")

        # 记录开始时间和初始化进程监控
        start_time = time.time()
        process = psutil.Process(os.getpid())

        for epoch in pbar:
            train_loss_curve = train(model, device, train_loader, optimizer, dataset.task_type)
            train_losses.append(train_loss_curve)
            valid_perf, val_loss, val_loss_curve = eval(model, device, test_loader, evaluator, dataset.task_type)
            val_losses.append(val_loss_curve)
            if maximize:
                if valid_perf[args.eval_metric] >= early_stopper.val_acc:
                    test_perf_at_best, test_loss_at_best, test_loss_curve = eval(model, device, valid_loader, evaluator,
                                                                                 dataset.task_type)
                    test_losses.append(test_loss_curve)

            else:
                if valid_perf[args.eval_metric] <= early_stopper.val_acc:
                    test_perf_at_best, test_loss_at_best, test_loss_curve = eval(model, device, valid_loader, evaluator,
                                                                                 dataset.task_type)
                    test_losses.append(test_loss_curve)

            if scheduler is not None:
                scheduler.step(valid_perf[args.eval_metric])
                if args.dataset == 'ZINC' and optimizer.param_groups[0]['lr'] <= 1e-5:
                    break
            if early_stopper.stop(epoch, val_loss, valid_perf[args.eval_metric], model=model):
                break

            # nni.report_intermediate_result(test_perf_at_best[args.eval_metric])
            pbar.set_description(
                f"{seed + 1}/{n_seeds}  Epoch {epoch + 1} Val loss {round(val_loss, 3)} Val perf {valid_perf[args.eval_metric]:.3f} Best Val Loss {early_stopper.val_loss:0.3f} Best Val Perf {early_stopper.val_acc:0.3f} Test Perf @ Best {test_perf_at_best[args.eval_metric]:0.3f}")


        # 程序结束后输出时间和内存
        end_time = time.time()

        test_perf_at_end, test_loss_at_end, _ = eval(model, device, test_loader, evaluator, dataset.task_type)
        train_perf_at_end, train_loss_at_end, _ = eval(model, device, train_loader, evaluator, dataset.task_type)

        train_perfs_at_end.append(train_perf_at_end[args.eval_metric])
        val_perfs_at_end.append(valid_perf[args.eval_metric])
        test_perfs_at_end.append(test_perf_at_end[args.eval_metric])
        val_perfs_at_best.append(early_stopper.val_acc)
        test_perfs_at_best.append(test_perf_at_best[args.eval_metric])
        best_epochs.append(early_stopper.best_epoch + 1)

        msg = (
            f'============= Results {seed + 1}/{n_seeds}=============\n'
            f'Dataset:        {args.dataset}\n'
            f'-------------  Best epoch ------------------\n'
            f'Best epoch:     {early_stopper.best_epoch + 1}\n'
            f'Validation:     {early_stopper.val_acc:0.4f}    Seed Average: {np.mean(val_perfs_at_best):0.4f} +/- {np.std(val_perfs_at_best):0.4f}\n'
            f'Test:           {test_perf_at_best[args.eval_metric]:0.4f}    Seed Average: {np.mean(test_perfs_at_best):0.4f} +/- {np.std(test_perfs_at_best):0.4f}\n'
            '--------------- Last epoch -----------------\n'
            f'Train:          {train_perf_at_end[args.eval_metric]:0.4f}    Seed Average: {np.mean(train_perfs_at_end):0.4f} +/- {np.std(train_perfs_at_end):0.4f}\n'
            f'Validation:     {valid_perf[args.eval_metric]:0.4f}    Seed Average: {np.mean(val_perfs_at_end):0.4f} +/- {np.std(val_perfs_at_end):0.4f}\n'
            f'Test:           {test_perf_at_end[args.eval_metric]:0.4f}    Seed Average: {np.mean(test_perfs_at_end):0.4f} +/- {np.std(test_perfs_at_end):0.4f}\n'
            '-------------------------------------------\n\n')

        log.print_and_log(msg)
        fs.append(test_perf_at_best[args.eval_metric])

    results = {
        "perfs_at_end": {"train": np.asarray(train_perfs_at_end), "val": np.asarray(val_perfs_at_end),
                         "test": np.asarray(test_perfs_at_end)},
        "perfs_at_best": {"val": np.asarray(val_perfs_at_best), "test": np.asarray(test_perfs_at_best)},
        "best_epochs": np.asarray(best_epochs),
        "train_loss_curve": np.asarray(train_losses),
        "val_loss_curve": np.asarray(val_losses),
        "test_loss_curve": np.asarray(test_losses)}

    with open(os.path.join("results", args.dataset, args.dataset + "_" + "results_" + program_name + ".pkl"),
              "wb") as f:
        pickle.dump(results, f)

    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    memory_used_mb = process.memory_info().rss / (1024 ** 2)

    # 输出到日志和控制台
    log.print_and_log(f"\n程序总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    log.print_and_log(f"程序最大内存占用: {memory_used_mb:.2f} MB")


if __name__ == "__main__":
    main()
