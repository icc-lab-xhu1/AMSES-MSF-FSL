import os
import argparse
import random
from src.eval_methods import *
import pickle as pk
from args import get_parser
import numpy as np
from sklearn import metrics
from torch.backends import cudnn
from utils.utils import *
# from data_factory.data_loader import *

from choosemodel_train import Solver

from pretrain_weak import Pretrain
import pandas as pd
from tqdm import tqdm
# from solver_woGP import Solver
from sample_train_ticket import chouyang
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from sknetwork.ranking import PageRank
import networkx as nx
from tabulate import tabulate
def my_acc(scoreList, rightOne, n=None):
    """Accuracy for Root Cause Analysis with multiple causes.
    Refined from the Acc metric in TBAC paper.
    """
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s


def prCal(scoreList, prk, rightOne):
    """计算scoreList的prk值

    Params:
        scoreList: list of tuple (node, score)
        prk: the top n nodes to consider
        rightOne: ground truth nodes
    """
    prkSum = 0
    for k in range(min(prk, len(scoreList))):
        if scoreList[k][0] in rightOne:
            prkSum = prkSum + 1
    denominator = min(len(rightOne), prk)
    return prkSum / denominator


def pr_stat(scoreList, rightOne, k=5):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne)
    return prkS


def print_prk_acc(prkS, acc):
    headers = ['PR@{}'.format(i + 1) for i in range(len(prkS))] + ['PR@Avg', 'Acc']
    data = prkS + [np.mean(prkS)]
    data.append(acc)
    print(tabulate([data], headers=headers, floatfmt="#06.4f"))

def evaluate(score_list, true_root_cause):
    acc = my_acc(score_list, true_root_cause)
    prks = pr_stat(score_list, true_root_cause, k=5)
    print_prk_acc(prks, acc)


def str2bool(v):
    return v.lower() in ('true')




def main(config):
    #random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    dict_values = {'cpu': 1, 'tx': 2, 'rx': 3}
    # model_values = {'WPS': 1, 'LSTM-VAE': 2, 'MHAR': 3}


    solver = Solver(vars(config))


    if config.mode == "pretrain":
        result = {}

        print("======================PRE TRAIN ======================")
        data_name = ['cpu', 'tx', 'rx']

        for name in data_name:
            wps, lstm, mhar, mad, usad, dag, mtad, omni, cae, msc = [], [], [], [], [], [], [], [], [], []

            pretrain = Pretrain(vars(config))
            for m in range(5):
                epoch = 10
                train, test_cpu, pre_label, input_cpu, values_1 = chouyang(name)
                i = len(train)/6
                low = i * m
                high = i * (m+1)

                random_train = random.randint(int(low), int(high))
                print(random_train)
                train = train[random_train: random_train + 300, :]
                dict_cpu = pretrain.train(train, test_cpu, pre_label, input_cpu, epoch, values_1, str(m))
                wps.append([v for k, v in dict_cpu.items() if k == 'WPS'])
                mhar.append([v for k, v in dict_cpu.items() if k == 'MHAR'])
                mad.append([v for k, v in dict_cpu.items() if k == 'MAD_GAN'])
                usad.append([v for k, v in dict_cpu.items() if k == 'USAD'])
                dag.append([v for k, v in dict_cpu.items() if k == 'DAGMM'])
                mtad.append([v for k, v in dict_cpu.items() if k == 'MTAD'])
                omni.append([v for k, v in dict_cpu.items() if k == 'Omni'])
                cae.append([v for k, v in dict_cpu.items() if k == 'CAE_M'])
                msc.append([v for k, v in dict_cpu.items() if k == 'MSCRED'])




            print(wps)
            # print(lstm)
            print(mhar)
            print(mad)
            print(usad)
            print(dag)
            print(mtad)
            print(omni)
            print(cae)
            print(msc)

            wps = np.mean(np.array(wps))

            mhar = np.mean(np.array(mhar))
            mad = np.mean(np.array(mad))
            usad = np.mean(np.array(usad))
            dag = np.mean(np.array(dag))
            mtad = np.mean(np.array(mtad))
            omni = np.mean(np.array(omni))
            cae = np.mean(np.array(cae))
            msc = np.mean(np.array(msc))




            dict = {'WPS': wps,  'MHAR': mhar, 'MAD_GAN': mad, 'USAD': usad, 'DAGMM': dag,
                    'MTAD': mtad, 'Omni': omni, 'CAE_M': cae, 'MSCRED': msc}
            print(name, dict)
            max_cpu = sorted(dict.items(), key=lambda item: item[1], reverse=True)[0:2]
            result.update({name: max_cpu})



        print(result)
        with open(r'./result/'+config.dataset+'_result_rca', 'wb') as f:
            pk.dump(result, f)



    if config.mode == 'train':

        # print("======================TRAIN MODE======================")

        with open(r'./result/' + config.dataset + '_result_rca', 'rb+') as f:
            result = pk.load(f)
        f.close()

        if config.scheme == 'ascending':
            result = sorted(result.items(), key=lambda item: item[1][0][1] + item[1][1][1], reverse=False)
        else:
            result = sorted(result.items(), key=lambda item: item[1][0][1] + item[1][1][1], reverse=True)
        print(result)

        for i in range(len(result)):
            data_name = result[i][0]
            train_model_1 = result[i][1][0][0]
            solver.train(train_model_1, data_name)
        for i in range(len(result)):
            data_name = result[i][0]
            train_model_2 = result[i][1][1][0]
            solver.train(train_model_2, data_name)


    elif config.mode == 'test':

        with open(r'./result/' + config.dataset + '_result_rca', 'rb+') as f:
            result = pk.load(f)
        f.close()

        if config.scheme == 'ascending':
            result = sorted(result.items(), key=lambda item: item[1][0][1] + item[1][1][1], reverse=False)
        else:
            result = sorted(result.items(), key=lambda item: item[1][0][1] + item[1][1][1], reverse=True)

        print(result)
        name = config.data_path.split('/')[1]
        data = pd.read_csv(config.data_path + "/" + name + "_test.csv")
        # print(data)
        # test_1 = data
        data['class'] = 0
        for i in range(len(result)):
            data_name = result[i][0]
            train_model_1 = result[i][1][0][0]
            train_model_2 = result[i][1][1][0]

            if config.alp == 1:
                name = [train_model_1, train_model_1]
            elif config.alp == 0:
                name = [train_model_2, train_model_2]
            else:
                name = [train_model_1, train_model_2]
            test_1 = data[data['class'] == 0]


            values_1 = [v for k, v in dict_values.items() if k == data_name][0]
            test = test_1
            for m in test.columns:
                if data_name not in m:
                    test_1 = test_1.drop(m, axis=1)
            test_1 = np.array(test_1)
            label_1 = np.array(test['label'].values)
            print('test shape:', test_1.shape)

            score_1, score_2, thred_1, thred_2 = solver.test(name, test_1, label_1, values_1, data_name)
            label = label_1[config.win_size: -1]
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)

            print('alpha:', config.alp)
            print('score_2', np.sum(score_2))
            print('score_1', np.sum(score_1))
    #         # score_2 = 0

            test_score = config.alp * score_1 + (1-config.alp) * score_2
            thred = config.alp * thred_1 + (1-config.alp) * thred_2
            with open(r'./thred/' + config.dataset + '_' + data_name + str(config.anomaly) + '.pkl', 'wb') as f:
                pk.dump(thred, f)
            print('thred:', thred)

            print('test score', test_score.shape)
            print('label', two_label1.shape)
            bf_eval_1 = bf_search(test_score, two_label1, start=0.001, end=1, step_num=150, verbose=False)

            print(data_name, bf_eval_1)
            pred_1 = bf_eval_1['pre']
            print(pred_1)
            for i, index in enumerate(test.index[config.win_size:-1]):
                if pred_1[i]:
                    data.loc[index, 'class'] = values_1
                else:
                    data.loc[index, 'class'] = 0

        rca_data = data[data['class'] != 0]
        rca_data.to_csv(r'./data_result/' + config.dataset + '_' + str(config.anomaly) + '.csv')
        print('准确率:', metrics.accuracy_score(data['label'].values, data['class'].values))  # 预测准确率输出

        print('宏平均精确率:', metrics.precision_score(data['label'].values, data['class'].values, average='macro'))  # 预测宏平均精确率输出
        print('宏平均召回率:', metrics.recall_score(data['label'].values, data['class'].values, average='macro'))  # 预测宏平均召回率输出
        print('宏平均F1-score:', metrics.f1_score(data['label'].values, data['class'].values, labels=[0, 1, 2, 3],
                                               average='macro'))  # 预测宏平均f1-score输出

        print('微平均精确率:',metrics.precision_score(data['label'].values, data['class'].values, average='micro'))  # 预测微平均精确率输出
        print('微平均召回率:', metrics.recall_score(data['label'].values, data['class'].values, average='micro'))  # 预测微平均召回率输出
        print('微平均F1-score:', metrics.f1_score(data['label'].values, data['class'].values, labels=[0, 1, 2, 3],
                                               average='micro'))  # 预测微平均f1-score输出
        print('加权平均精确率:',metrics.precision_score(data['label'].values, data['class'].values, average='weighted'))  # 预测加权平均精确率输出
        print('加权平均召回率:', metrics.recall_score(data['label'].values, data['class'].values, average='micro'))  # 预测加权平均召回率输出
        print('加权平均F1-score:', metrics.f1_score(data['label'].values, data['class'].values, labels=[0, 1, 2, 3], average='weighted'))  # 预测加权平均f1-score输出

        print('混淆矩阵输出:\n', metrics.confusion_matrix(data['label'].values, data['class'].values, labels=[0, 1, 2, 3]))  # 混淆矩阵输出
        print('分类报告:\n', metrics.classification_report(data['label'].values, data['class'].values, labels=[0, 1, 2, 3]))  # 分类报告输出



    elif config.mode == 'pre_rca':

        set_name = config.data_path.split('/')[1]
        pretrain = Pretrain(vars(config))
        data_name = ['cpu', 'tx', 'rx']
        result_rca = {}
        train_data = pd.read_csv(config.data_path + "/" + config.dataset + "_train.csv")
        for name in data_name:
            rca_pc, rca_ges, rca_lin = [], [], []
            for m in range(10):
                values = [v for k, v in dict_values.items() if k == name][0]
                rca_data = train_data[train_data['label'] == values]

                i = len(rca_data) / 11
                low = i * m
                high = i * (m + 1)

                rca_train = random.randint(int(low), int(high))
                print(rca_train)
                rca_data = rca_data.iloc[rca_train: rca_train + 30, :]
                data_rca = rca_data
                if values == 1:
                    for r in rca_data.columns:
                        if 'cpu' not in r:
                            data_rca = data_rca.drop(r, axis=1)
                elif values == 2:
                    for r in rca_data.columns:
                        if 'tx' not in r:
                            data_rca = data_rca.drop(r, axis=1)
                else:
                    for r in rca_data.columns:
                        if 'rx' not in r:
                            data_rca = data_rca.drop(r, axis=1)
                print(data_rca)

                col = list(data_rca.columns)
                true_root_cause = [col.index(set_name + '_' + name)]
                print('true_root_cause:', true_root_cause)
                rca_dict = pretrain.rca(data_rca, true_root_cause)
                print("======================"+str(m)+" RCA TRAIN ======================")

                rca_pc.append([v for k, v in rca_dict.items() if k == 'PC'])
                rca_ges.append([v for k, v in rca_dict.items() if k == 'GES'])
                rca_lin.append([v for k, v in rca_dict.items() if k == 'Lingam'])
            rca_pc = np.mean(np.array(rca_pc))
            rca_ges = np.mean(np.array(rca_ges))
            rca_lin = np.mean(np.array(rca_lin))
            dict_r = {'PC': rca_pc, 'GES': rca_ges, 'Lingam': rca_lin}
            print(name, dict_r)
            max_rca = sorted(dict_r.items(), key=lambda item: item[1], reverse=True)[0]
            print('max_rca', max_rca)
            result_rca.update({name: max_rca})
        print(result_rca)
        with open(r'./result/' + config.dataset + '_rca_result', 'wb') as f:
            pk.dump(result_rca, f)
    elif config.mode == 'rca':
        print("======================RCA TEST ======================")
        with open(r'./result/' + config.dataset + '_rca_result', 'rb+') as f:
            result_rca  = pk.load(f)
        f.close()
        rca = 0.5

        result = sorted(result_rca.items(), key=lambda item: item[1], reverse=True)
        print(result)
        test_data = pd.read_csv(r'./data_result/' + config.dataset + '_' + str(config.anomaly) + '.csv')

        for i in range(len(result)):
            key = result[i][0]
            with open(r'./thred/' + config.dataset + '_' + key + str(config.anomaly) + '.pkl', 'rb+') as f:
                thred = pk.load(f)
            f.close()

            meth = result[i][1][0]
            values = [v for k, v in dict_values.items() if k == key][0]
            data_rca = test_data[test_data['class'] == values]
            data = data_rca
            if values == 1:
                for m in data.columns:
                    if 'cpu' not in m:
                        data_rca = data_rca.drop(m, axis=1)
            elif values == 2:
                for m in data.columns:
                    if 'tx' not in m:
                        data_rca = data_rca.drop(m, axis=1)
            else:
                for m in data.columns:
                    if 'rx' not in m:
                        data_rca = data_rca.drop(m, axis=1)
            col = list(data_rca.columns)
            true_root_cause = [col.index(config.dataset + '_' + key)]
            print('true_root_cause:', true_root_cause)
            # print(data_rca)
            if result[i][1][0] == 'Lingam':

                X = np.array(data_rca)


                model = lingam.ICALiNGAM()
                model.fit(X)


                print('LiNGAM result')
                print(model.causal_order_)  # the later virable is unable to cause the former virable
                # print(model.adjacency_matrix_)
                # G = nx.DiGraph()
                adj = model.adjacency_matrix_.T
                pagerank = PageRank()
                scores = pagerank.fit_transform(adj.T)
                print(scores)

                score_dict = {}
                for i, s in enumerate(scores):
                    score_dict[i] = s
                rank = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

                print('------------' + str(key) + ' Lingam -------------')
                evaluate(rank, true_root_cause)

            elif result[i][1][0] == 'GES':

                X = data_rca


                X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
                X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
                maxP = 5  # maximum number of parents when searching the graph
                parameters = {'kfold': 10, 'lambda': 0.01}

                Record = ges(X, 'local_score_CV_general', maxP=maxP, parameters=parameters)
                adj = Record['G'].graph

                # Change the adj to graph
                G = nx.DiGraph()
                for i in range(len(adj)):
                    for j in range(len(adj)):
                        if adj[i, j] == -1:
                            G.add_edge(i, j)
                        if adj[i, j] == 1:
                            G.add_edge(j, i)
                nodes = sorted(G.nodes())
                print(nodes)
                adj = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
                print(adj)
                pos = nx.circular_layout(G)
                nx.draw(G, pos=pos, with_labels=True)
                # PageRank
                pagerank = PageRank()
                scores = pagerank.fit_transform(adj.T)
                print(scores)
                # cmap = plt.cm.coolwarm

                dict_s = {}
                for inx, i in enumerate(nodes):
                    dict_s.update({i: scores[inx]})
                rank = sorted(dict_s.items(), key=lambda item: item[1], reverse=True)
                print(sorted(dict_s.items(), key=lambda item:item[1], reverse=True))
                print(meth, key)
                print('------------' + str(key) + ' GES -------------')
                evaluate(rank, true_root_cause)

            else:

                X = data_rca
                cg = pc(X.to_numpy(), 0.05, fisherz, False, 0, -1)
                adj = cg.G.graph

                print('PC result')
                print(adj)

                # Change the adj to graph
                G = nx.DiGraph()
                for i in range(len(adj)):
                    for j in range(len(adj)):
                        if adj[i, j] == -1:
                            G.add_edge(i, j)
                        if adj[i, j] == 1:
                            G.add_edge(j, i)
                nodes = sorted(G.nodes())
                print(nodes)
                adj = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
                pos = nx.circular_layout(G)
                print(adj)
                nx.draw(G, pos=pos, with_labels=True)
                pagerank = PageRank()
                scores = pagerank.fit_transform(adj.T)
                print(scores)
                dict_s = {}
                for inx, i in enumerate(nodes):
                    dict_s.update({i: scores[inx]})
                rank = sorted(dict_s.items(), key=lambda item: item[1], reverse=True)
                #     print(sorted(dict_s.items(), key=lambda item:item[1], reverse=True))
                print(meth, key)
                print('------------' + str(key) + ' PC -------------')
                evaluate(rank, true_root_cause)


if __name__ == '__main__':

    parser = get_parser()
    config = parser.parse_args()

    args = vars(config)


    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    for i in range(1):
        config.seed = 3
        main(config)
