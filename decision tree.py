# https://github.com/Erikfather/Decision_tree-python/blob/master/tree.py
# https://zhuanlan.zhihu.com/p/65304798
import numpy as np
import data_process
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time

class dcs_tree(object):

    def __init__(self, impurity='ent_gr'):
        self.impurity = impurity
        self.dcs_tree = dict()
        self.weights_mis = None

    def train(self, ftr: np.ndarray, tgt: np.ndarray):
        feature = np.array(ftr)
        target = np.array(tgt)
        ftr_indices = set(range(0, feature.shape[-1]))
        #pos = np.where(feature!=-1)
        # processing missing values
        #pos_mis_values = np.where(feature == -1)
        #line_values = list(set([item for item in pos[0]]))
        #line_mis_values = list(set([item for item in pos_mis_values[0]]))
        self.weights = np.ones(feature.shape[0])#{line:1 for line in line_mis_values}

        self.dcs_tree = self.induct_tree(feature, target, ftr_indices, self.weights)

    def classify(self, ftr: np.ndarray)->list:
        ftr = np.array(ftr)
        labels = list()
        for record in ftr:
            tree = self.dcs_tree
            idx_ftr = list(tree["tree"].keys())[0]
            #print(tree["tree"].keys())
            #print(idx_ftr)
            idx_ftr = int(idx_ftr)
            #print(idx_ftr)
            p=1
            final_p = {0:0,1:0} #TODO
            #final_p = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            final_p = dcs_tree.search_tree(record,tree,idx_ftr,p,final_p)
            #print("final_p",final_p)
            '''while True:
                ftr_val = record[idx_ftr]
                if ftr_val == -1:
                    val_list = list(tree["tree"][str(idx_ftr)])
                    for val in val_list:
                        count_list = list()
                        p_list = list()
                        count_total = 0
                        if type(tree["tree"][str(idx_ftr)][val]).__name__ == 'dict':
                            tree = tree["tree"][str(idx_ftr)][val]
                            count = tree["count"][str(idx_ftr)][val]
                            count_list.append(count*1.0)
                        for i in count_list:
                            count_total += i
                        p_list = [i/count_total for i in count_list]

                if type(tree["tree"][str(idx_ftr)][ftr_val]).__name__ == 'dict':
                    tree = tree["tree"][str(idx_ftr)][ftr_val]
                    idx_ftr = list(tree["tree"].keys())[0]
                    idx_ftr = int(idx_ftr)
                else:

                    labels.append(tree[str(idx_ftr)][ftr_val])
                    break'''
            label = max(final_p,key = final_p.get)
            labels.append(label)
        return labels

    def induct_tree(self, ftr: np.ndarray, tgt: np.ndarray, ftr_set: set,weights:np.ndarray)->dict:
        #if np.unique(tgt).shape == tgt.shape:
            #return tgt[0]
        if len(ftr_set) == 1 or np.unique(ftr).shape == ftr.shape:  # stop criteria
            return dcs_tree.majorityVote(tgt)

        idx_best, gr = self.best_split(ftr, tgt, ftr_set, weights)  # TODO
        if gr < 0:
            return dcs_tree.majorityVote(tgt)  # stop criteria
        node = {"tree":{str(idx_best):{}},"count":{str(idx_best):{}}}

        ftr_best = ftr[:, idx_best]
        idx_pure = np.where(ftr_best!=-1)
        ftr_best_pure = ftr_best[idx_pure]  # no missing value
        ftr_cls = np.unique(ftr_best_pure)
        n_true = len(idx_pure[0])
        for c in ftr_cls:
            pos = np.where((ftr_best == c)|(ftr_best == -1))  # include missing value

            missing_list = np.where(ftr_best == -1)
            #true_list = np.where(ftr_best == c)
            n_truec = len(pos[0])-len(missing_list[0])
            adjust_r = n_truec*1.0/n_true
            weights[missing_list] = weights[missing_list]*adjust_r
            count = np.sum(weights[pos])
            node["count"][str(idx_best)][c] = count
            node["tree"][str(idx_best)][c] = self.induct_tree(ftr[pos], tgt[pos],
                                                      ftr_set-{idx_best}, weights[pos])

        return node

    def best_split(self, ftr: np.ndarray, tgt: np.ndarray, idx: set, weights:np.ndarray)->object:
        index_best = 0
        if self.impurity == 'ent_gr':  # entropy gain ratio

            g_max = -1 * np.inf
            for i in idx:
                gain = self.info_gain(ftr[:, i], tgt, weights)  # TODO missing list
                if gain > g_max:
                    g_max = gain
                    index_best = i

        return index_best, g_max

    @staticmethod
    def majorityVote(tgt: np.ndarray):

        votes = sorted([(np.sum(tgt == i), i, idx) for idx,i in enumerate(np.unique(tgt))], key=lambda x: x[0],reverse = True)#[-1][-1]
        #node = {"tree":{}, "count":{}}
        node = {"tree": [], "count": []}
        for item in votes:
            node["tree"].append(item[-2])
            node["count"].append(item[0])

        return node

    @staticmethod
    def compute_ent(tgt: np.ndarray, weights:np.ndarray)->float:
        entropy = 0.0
        #n_true = np.sum(weights)
        #n_m = np.sum(weights_m)
        n_m = np.sum(weights)
        cls = np.unique(tgt)
        for c in cls:
            n_truec = np.sum(weights[np.where(tgt==c)])
            #n_m = tgt_m[tgt_m==c].size
            #weights_mc = weights_m[np.where(tgt_m==c)]
            #n_mc = np.sum(weights_mc)
            p = 1.0*n_truec/n_m
            #p = tgt[tgt == c].size/n
            entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def info_gain(ftr_i: np.ndarray, tgt: np.ndarray,weights:np.ndarray)->float:

        entropy = dcs_tree.compute_ent(tgt, weights)

        nw = np.sum(weights)
        vals = np.unique(ftr_i)
        nf = np.sum(weights[np.where(ftr_i!=-1)])
        for v in vals:
            id = np.where((ftr_i == v) & (ftr_i!=-1))
            entropy_child = dcs_tree.compute_ent(tgt[id], weights[id])
            n_m = np.sum(weights[id])
            entropy = entropy-1.0*n_m/nf * entropy_child
        entropy = 1.0*nf/nw * entropy
        #id_m = np.where(ftr_i)
        #tgt_m = tgt[id_m]

        return entropy

    '''@staticmethod  # TODO
    def compute_gr(ftr_i: np.ndarray, tgt: np.ndarray, ftr_mi: np.ndarray, tgt_m:np.ndarray, weights: np.ndarray)->float:

        split_info = 0.0
        n = np.sum(weights)
        n = tgt.size
        n_m = np.sum(weights_i)
        vals = np.unique(ftr_i)
        for v in vals:
            id = np.where(ftr_i == v)
            ratio = tgt[id].size/n
            adjust_ratio =
            # split_info -= tgt[id].size/n * np.log2(tgt[id].size/n)
        if split_info == 0.0:
            return 0.0
        gr = dcs_tree.info_gain(ftr_i, tgt) / split_info

        return gr'''

    @staticmethod
    def search_tree(record, tree, idx:int,p, final_p:dict):
        ftr_val = record[idx]
        if ftr_val not in tree["tree"][str(idx)].keys():
            ftr_val = -1
        if ftr_val == -1:

            val_list = list(tree["tree"][str(idx)].keys())
            count_list = [tree["count"][str(idx)][val] for val in val_list]
            total = sum(count_list)
            p_list = [1.0*c/total*p for c in count_list]
            #final_p = {0:0,1:0}
            for val, pi in zip(val_list,p_list):
                newtree = tree["tree"][str(idx)][val]
                if type(newtree["tree"]).__name__ == 'dict':
                    newidx = list(newtree["tree"].keys())[0]
                    newidx = int(newidx)
                    final_p = dcs_tree.search_tree(record,newtree,newidx, pi, final_p)
                else:
                    #ranklist = list(newtree["tree"].keys())
                    tgtlist = newtree["tree"]
                    countlist = newtree["count"]
                    p_dict = {0:0,1:0} # TODO
                    #p_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                    total_c = sum(countlist)
                    for tgt,c in zip(tgtlist,countlist):
                        p_dict[tgt] = 1.0*c/total_c*pi
                        final_p[tgt] += p_dict[tgt]
            return final_p
        else:
            node = tree["tree"][str(idx)][ftr_val]
            if type(node["tree"]).__name__ == 'dict':
                idx = list(node["tree"].keys())[0]
                idx = int(idx)

                return dcs_tree.search_tree(record, node, idx, p, final_p)
            else:
                tgtlist = node["tree"]
                countlist = node["count"]
                p_dict = {0: 0, 1: 0} # TODO
                #p_dict = {0: 0, 1: 0, 2:0,3:0,4:0}
                total_c = sum(countlist)
                for tgt, c in zip(tgtlist, countlist):
                    p_dict[tgt] = 1.0 * c / total_c*p
                    final_p[tgt] += p_dict[tgt]
                return final_p






if __name__ == '__main__':
    #data = np.random.randint(0,5, size=(10,5))
    #print(data)
    #model = dcs_tree()
    #data[0,0] = -1
    #data[8,0] = -1
    #data[8, 3] = -1
    #model.train(data[:6,:4],data[:6,4])
    #print(model.dcs_tree)
    #labels = model.classify(data[6:,:4])
    #print(labels)
    start_d = time.time()
    data = data_process.read_data(data_process.filename)
    data = data_process.preprocess(data)
    data_tt = data_process.read_data(data_process.filename_test)
    data_tt = data_process.preprocess(data_tt)
    end_d = time.time()
    print("time for data processing:", end_d-start_d)
    model = dcs_tree()

    k = 10
    kf = KFold(n_splits=k)
    acc_list = []
    prcs_list = []
    r_list = []
    f1_list = []
    acc_list_t = []
    prcs_list_t = []
    r_list_t = []
    f1_list_t = []
    for train_idx, test_idx in kf.split(data):
        f_tr, f_tt = data[:,:-1][train_idx], data[:,:-1][test_idx]
        label_tr, label_tt = data[:,-1][train_idx], data[:,-1][test_idx]

        start = time.time()
        model.train(f_tr, label_tr)
        end = time.time()
        print("time for training:", end-start)
        start_p = time.time()
        labels = model.classify(f_tt)
        end_p = time.time()
        print("time for predicting:", end_p-start_p)

        acc = accuracy_score(label_tt, labels)
        prcs = precision_score(label_tt,labels)
        recall = recall_score(label_tt,labels)
        f1 = f1_score(label_tt,labels)
        acc_list.append(acc)
        prcs_list.append(prcs)
        r_list.append(recall)
        f1_list.append(f1)

        # print("time for training:", end - start)
        start_tp = time.time()
        labels_t = model.classify(data_tt[:,:-1])
        end_tp = time.time()
        print("time for test set predicting:", end_tp - start_tp)

        acc_t = accuracy_score(data_tt[:,-1], labels_t)
        prcs_t = precision_score(data_tt[:,-1], labels_t)
        recall_t = recall_score(data_tt[:,-1], labels_t)
        f1_t = f1_score(data_tt[:,-1], labels_t)
        acc_list_t.append(acc_t)
        prcs_list_t.append(prcs_t)
        r_list_t.append(recall_t)
        f1_list_t.append(f1_t)



    print("accuracy:", acc_list)
    print("recall:", r_list)
    print("precision:", prcs_list)
    print("f1:", f1_list)
    print("average_a:", 1.0*sum(acc_list)/k)
    print("average_r:", 1.0*sum(r_list)/k)
    print("average_p:", 1.0*sum(prcs_list)/k)
    print("average_f1:", 1.0*sum(f1_list)/k)

    print("accuracy:", acc_list_t)
    print("recall:", r_list_t)
    print("precision:", prcs_list_t)
    print("f1:", f1_list_t)
    print("average_a:", 1.0 * sum(acc_list_t) / k)
    print("average_r:", 1.0 * sum(r_list_t) / k)
    print("average_p:", 1.0 * sum(prcs_list_t) / k)
    print("average_f1:", 1.0 * sum(f1_list_t) / k)

















