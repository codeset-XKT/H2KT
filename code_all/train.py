import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from train_utils import Logger
from sklearn import metrics
from loader import load_data
from model import H2KT
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据参数配置
def parse_args():
    data_set = 'JunYi'
    parser = argparse.ArgumentParser()

    ques_num_dict = {
        'ASSIST17': 1146,
        'EdNet': 11818,
        'Statics2011': 633,
        'JunYi': 662
    }

    # 数据参数配置
    parser.add_argument("--batch_size", type=int,
                        default=32)
    parser.add_argument("--min_seq_len", type=int,
                        default=10)
    parser.add_argument("--max_seq_len", type=int,
                        default=1000)
    parser.add_argument('--device', type=str,
                        default="cuda")
    parser.add_argument("--input", type=str,
                        default='all_feature')
    parser.add_argument("--data_path", type=str,
                        default=r"..\data_all")
    parser.add_argument("--data_set", type=str,
                        default=data_set)
    parser.add_argument("--ques_num", type=int,
                        default=ques_num_dict[data_set])
    parser.add_argument('--save_dir', type=str,
                        default='./result/{0}'.format(data_set),
                        help='the dir which save results')
    parser.add_argument('--log_file', type=str,
                        default='logs.txt',
                        help='the name of logs file')
    parser.add_argument('--result_file', type=str,
                        default='tunings.txt',
                        help='the name of results file')
    parser.add_argument('--remark', type=str,
                        default='', help='remark the experiment')
    parser.add_argument("--patience", type=int,
                        default=15)

    # 模型参数配置
    parser.add_argument("--epochs", type=int,
                        default=200)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--l2_weight", type=float,
                        default=1e-6)
    parser.add_argument("--embed_dim", type=int,
                        default=64)
    parser.add_argument("--num_layer", type=int,
                        default=1)
    parser.add_argument("--subgraph_embed", type=bool,
                        default=True)

    return parser.parse_args()


args = parse_args()

def demo_train(args):
    print("Parameter configuration\n\n", args, "\n")
    logger = Logger(args)

    # Loading test data
    loader = load_data(args)
    model = H2KT(args)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.l2_weight)
    criterion = nn.BCELoss()
    for epoch in range(args.epochs):

        logger.epoch_increase()
        epoch_loss = 0
        for i, (
                seq_lens, pad_data, pad_answer, pad_index, pad_label, timestaps, attempts, answertimes) in enumerate(
            loader['train']):
            pad_predict = model(pad_data, pad_answer, pad_index, timestaps, attempts, answertimes)  # [bsz,seq_len]
            pack_predict = pack_padded_sequence(pad_predict, seq_lens.to("cpu"), enforce_sorted=True,
                                                batch_first=True)  # [bsz*seq_len]
            pack_label = pack_padded_sequence(pad_label, seq_lens.to("cpu"), enforce_sorted=True,
                                              batch_first=True)  # [bsz*seq_len]

            loss = criterion(pack_predict.data, pack_label.data)
            epoch_loss += loss.item()
            # print("epoch:",epoch+1,"batch",i+1,"loss",loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics_dict = evaluate(model, loader['train'])
        test_metrics_dict = evaluate(model, loader['test'])

        logger.one_epoch(epoch, train_metrics_dict, test_metrics_dict, model)

        if logger.is_stop():
            break
        print(epoch, "epoch_loss", epoch_loss)

    logger.one_run(args)


def evaluate(model, data):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pad_label, timestaps, attempts, answertimes in data:
        pad_predict = model(pad_data, pad_answer, pad_index, timestaps, attempts, answertimes)  # 运行模型
        pack_predict = pack_padded_sequence(pad_predict, seq_lens.to("cpu"), enforce_sorted=True,
                                            batch_first=True)  # [bsz*seq_len]
        pack_label = pack_padded_sequence(pad_label, seq_lens.to("cpu"), enforce_sorted=True,
                                          batch_first=True)  # [bsz*seq_len]

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)

    all_pred = torch.cat(pred_list, 0)
    all_target = torch.cat(true_list, 0)
    auc = metrics.roc_auc_score(all_target, all_pred)

    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    acc = metrics.accuracy_score(all_target, all_pred)

    model.train()
    return {'auc': auc, 'acc': acc}


demo_train(args)
