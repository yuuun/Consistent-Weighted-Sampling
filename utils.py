import numpy as np

class RandomNumberGenerator():
    def __init__(self, dim, n_sig):
        self.ran_r1 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_b = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]

def get_similarity(train_samples, test_samples, n_sig):
    similarity = []            # total similarity
    sorted_closest_idxs = []   # total index sorted with 
    for ts in test_samples:
        one_sim = []                # simlarity between one test dataset and rest train dataset
        for trs in train_samples:
            cnt = 0
            for t, tr in zip(ts, trs):
                if t[0] == tr[0] and t[1] == tr[1]:
                    cnt += 1
            one_sim.append(cnt / n_sig)

        similarity.append(one_sim)
        sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
        sorted_closest_idxs.append(sorted_similarity_idx)
        
    return similarity, sorted_closest_idxs

def calculate_acc(train_labels, test_labels, sorted_idxs):
    '''
    Accuracy is defined as follows,
    If the label of closest train data is same as train data, it is matched as correct, else, incorrect
    '''
    total_acc = 0.
    for label, idxs in zip(test_labels, sorted_idxs):
        if label == train_labels[idxs[0]]:
            total_acc += 1

    return total_acc / len(test_labels)

def calculate_prec(sorted_jacc, sorted_idxs, kList):
    '''
    We defined precision with jaccard similarity
    '''
    prec_list = []
    for k in kList:
        prec = []
        for jacc, idxs in zip(sorted_jacc, sorted_idxs):
            k_jacc, k_idxs = set(jacc[:k]), set(idxs[:k])
            n_match = len(k_jacc.intersection(k_idxs))
            prec.append(n_match / k)

        prec_list.append(sum(prec) / len(sorted_idxs))

    return prec_list


