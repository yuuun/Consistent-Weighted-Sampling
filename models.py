import numpy as np
import random
import time
import sys
import math
from utils import *


class GeneralizedJaccardSimliarity():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.calculate_jacc()
        self.get_acc()
    
    def calculate_jacc(self):
        start = time.time()
        self.sorted_closest_idxs = []
        for t_w, t_i in zip(self.test_weights, self.test_idxs):
            similarity = []
            for tr_w, tr_i in zip(self.train_weights, self.train_idxs):
                i, j, sum_min, sum_max = 0, 0, 0, 0
                while i < len(t_i) and j < len(tr_i):
                    if t_i[i] == tr_i[j]:
                        sum_min += min(t_w[i], tr_w[j])
                        sum_max += max(t_w[i], tr_w[j])
                        i += 1
                        j += 1
                    elif t_i[i] > tr_i[j]:
                        sum_max += tr_w[j]
                        j += 1
                    else:
                        sum_max += t_w[i]
                        i += 1

                while i < len(t_i):
                    sum_max += t_w[i]
                    i += 1
                while j < len(tr_i):
                    sum_max += tr_w[j]
                    j += 1
                    
                similarity.append(sum_min / sum_max)

            sorted_similarity_idx = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
            self.total_time = '{:.5f}s'.format(time.time() - start)

    def get_acc(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)


class ICWS():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.generate_random_number()
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def generate_random_number(self):
        self.ran_r1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for r1, r2, c1, c2, b in zip(self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.train_idxs[idx], self.train_weights[idx], r1, r2, c1, c2, b)
                self.train_samples[idx].append([k, yk])
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for idx in range(self.n_test):
            for r1, r2, c1, c2, b in zip(self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.test_idxs[idx], self.test_weights[idx], r1, r2, c1, c2, b)
                self.test_samples[idx].append([k, yk])

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, idxs, weights, r1, r2, c1, c2, b):
        minIdx, minVal, minY = 0, sys.maxsize, 0
        for idx, weight in zip(idxs, weights):
            if weight == 0:
                continue
            t = math.floor((math.log(weight) / -math.log(r1[idx] * r2[idx])) + b[idx])
            y = math.exp(-math.log(r1[idx] * r2[idx]) * (t - b[idx]))
            a = -math.log(c1[idx] * c2[idx]) * r1[idx] * r2[idx] / y
            
            if minVal > a:
                minVal = a
                minIdx = idx
                minY = y

        return minIdx, minY
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t[0] == tr[0] and t[1] == tr[1]:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)




class zero_ICWS():
    '''
    0-bit ICWS
    '''
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.generate_random_number()
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def generate_random_number(self):
        self.ran_r1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for r1, r2, c1, c2, b in zip(self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k = self.hashing(self.train_idxs[idx], self.train_weights[idx], r1, r2, c1, c2, b)
                self.train_samples[idx].append(k)
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for idx in range(self.n_test):
            for r1, r2, c1, c2, b in zip(self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k = self.hashing(self.test_idxs[idx], self.test_weights[idx], r1, r2, c1, c2, b)
                self.test_samples[idx].append(k)

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, idxs, weights, r1, r2, c1, c2, b):
        minIdx, minVal = 0, sys.maxsize
        for idx, weight in zip(idxs, weights):
            if weight == 0:
                continue
            t = math.floor((math.log(weight) / -math.log(r1[idx] * r2[idx])) + b[idx])
            y = math.exp(-math.log(r1[idx] * r2[idx]) * (t - b[idx]))
            a = -math.log(c1[idx] * c2[idx]) * r1[idx] * r2[idx] / y
            
            if minVal > a:
                minVal = a
                minIdx = idx

        return minIdx
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t == tr:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)



class CCWS():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.generate_random_number()
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def generate_random_number(self):
        self.ran_r = [[np.random.beta(2, 1)] * (self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for r, c1, c2, b in zip(self.ran_r, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.train_idxs[idx], self.train_weights[idx], r, c1, c2, b)
                self.train_samples[idx].append([k, yk])
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for idx in range(self.n_test):
            for r, c1, c2, b in zip(self.ran_r, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.test_idxs[idx], self.test_weights[idx], r, c1, c2, b)
                self.test_samples[idx].append([k, yk])

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, idxs, weights, r, c1, c2, b):
        minIdx, minVal, minY = 0, sys.maxsize, 0
        for idx, weight in zip(idxs, weights):
            if weight == 0:
                continue
            t = math.floor((math.log(weight) / r[idx]) + b[idx])
            y = r[idx] * (t - b[idx])
            a = -math.log(c1[idx] * c2[idx]) / y - 2 * r[idx] * (math.log(c1[idx] * c2[idx]))
            
            if minVal > a:
                minVal = a
                minIdx = idx
                minY = y

        return minIdx, minY
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t[0] == tr[0] and t[1] == tr[1]:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)



class PCWS():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.generate_random_number()
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def generate_random_number(self):
        self.ran_r1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for r1, r2, c, b in zip(self.ran_r1, self.ran_r2, self.ran_c, self.ran_b):
                k, yk = self.hashing(self.train_idxs[idx], self.train_weights[idx], r1, r2, c, b)
                self.train_samples[idx].append([k, yk])
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for idx in range(self.n_test):
            for r1, r2, c1, b in zip(self.ran_r1, self.ran_r2, self.ran_c, self.ran_b):
                k, yk = self.hashing(self.test_idxs[idx], self.test_weights[idx], r1, r2, c1, b)
                self.test_samples[idx].append([k, yk])

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, idxs, weights, r1, r2, c, b):
        minIdx, minVal, minY = 0, sys.maxsize, 0
        for idx, weight in zip(idxs, weights):
            if weight == 0:
                continue
            t = math.floor((math.log(weight) / -math.log(r1[idx] * r2[idx])) + b[idx])
            y = math.exp(-math.log(r1[idx] * r2[idx]) * (t - b[idx]))
            a = -math.log(c[idx]) * r1[idx] / y
            
            if minVal > a:
                minVal = a
                minIdx = idx
                minY = y

        return minIdx, minY
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t[0] == tr[0] and t[1] == tr[1]:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)


class I2CWS():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.generate_random_number()
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def generate_random_number(self):
        self.ran_r11 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r12 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r21 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_r22 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b1 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]
        self.ran_b2 = [np.random.uniform(0, 1, self.dim + 1) for _ in range(self.n_sig)]

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for r11, r12, r21, r22, c1, c2, b1, b2 in zip(self.ran_r11, self.ran_r12, self.ran_r21, self.ran_r22, self.ran_c1, self.ran_c2, self.ran_b1, self.ran_b2):
                k, yk = self.hashing(self.train_idxs[idx], self.train_weights[idx], r11, r12, r21, r22, c1, c2, b1, b2)
                self.train_samples[idx].append([k, yk])
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for idx in range(self.n_test):
            for r11, r12, r21, r22, c1, c2, b1, b2 in zip(self.ran_r11, self.ran_r12, self.ran_r21, self.ran_r22, self.ran_c1, self.ran_c2, self.ran_b1, self.ran_b2):
                k, yk = self.hashing(self.test_idxs[idx], self.test_weights[idx], r11, r12, r21, r22, c1, c2, b1, b2)
                self.test_samples[idx].append([k, yk])

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, idxs, weights, r11, r12, r21, r22, c1, c2, b1, b2):
        minIdx, minVal, minWeight = 0, sys.maxsize, 0
        for idx, weight in zip(idxs, weights):
            if weight == 0:
                continue
            t2 = math.floor((math.log(weight) / -math.log(r21[idx] * r22[idx])) + b2[idx])
            z = math.exp(-math.log(r21[idx] * r22[idx]) * (t2 - b2[idx] + 1))
            a = -math.log(c1[idx] * c2[idx]) / z
            
            if minVal > a:
                minVal = a
                minIdx = idx
                minWeight = weight
        
        tk1 = math.floor((math.log(minWeight) / -math.log(r11[minIdx] * r12[minIdx])) + b1[minIdx])
        y = math.exp(-math.log(r11[minIdx] * r12[minIdx]) * (tk1 - b1[minIdx]))

        return minIdx, y
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t[0] == tr[0] and t[1] == tr[1]:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)


class BCWS():
    def __init__(self, train_weights, train_idxs, train_labels, test_weights, test_idxs, test_labels, jacc, dim, n_sig, kList):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.train_labels = train_labels

        self.test_weights = test_weights
        self.test_idxs = test_idxs
        self.test_labels = test_labels
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)

        self.jacc = jacc
        
        self.dim = dim
        self.n_sig = n_sig  # number of samples(signature) to hash
        self.kList = kList

        self.select_bin()
        self.generate_random_number()
        self.splited_train_idxs, self.splited_train_weights = self.splitBins(self.train_idxs, train_weights)
        self.splited_test_idxs, self.splited_test_weights = self.splitBins(self.test_idxs, self.test_weights)
        self.generate_samples()
        self.get_similarity()
        self.get_performance()
    
    def select_bin(self):
        self.bins_list = [random.randrange(self.n_sig) for _ in range(self.n_sig)]
        ## selecting the order of re-randomizing idxs
        self.rerandomize_list = [random.sample(list(range(self.n_sig)), self.n_sig) for _ in range(self.n_sig)]
    
    def generate_random_number(self):
        self.numSigPerBins = self.dim // self.n_sig + 1
        self.ran_r1 = [np.random.uniform(0, 1, self.numSigPerBins) for _ in range(self.n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, self.numSigPerBins) for _ in range(self.n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, self.numSigPerBins) for _ in range(self.n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, self.numSigPerBins) for _ in range(self.n_sig)]
        self.ran_b = [np.random.uniform(0, 1, self.numSigPerBins) for _ in range(self.n_sig)]
    
    def splitBins(self, idxs_list, weights_list):
        splited_idxs = [[] for _ in range(len(idxs_list))]
        splited_weights = [[] for _ in range(len(weights_list))]
        for i in range(len(idxs_list)):
            idxs, weights = idxs_list[i], weights_list[i]
            s, e = 0, 0
            tmp_idxs = []
            tmp_weights = []
            k = 1
            while k <= self.n_sig:
                eidx = self.dim * k // self.n_sig
                while e < len(idxs) and idxs[e] <= eidx:
                    e += 1
                tmp_idxs.append(idxs[s:e])
                tmp_weights.append(weights[s:e])
                s = e
                k += 1
            splited_idxs[i] = tmp_idxs
            splited_weights[i] = tmp_weights
        return splited_idxs, splited_weights

    def generate_samples(self):
        start = time.time()
        self.train_samples = [[] for _ in range(self.n_train)]
        for idx in range(self.n_train):
            for num_bin, r1, r2, c1, c2, b in zip(self.bins_list, self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.splited_train_idxs[idx], self.splited_train_weights[idx], num_bin, r1, r2, c1, c2, b)
                self.train_samples[idx].append([k, yk])
        
        self.test_samples = [[] for _ in range(self.n_test)]
        for num_bin, r1, r2, c1, c2, b in zip(self.bins_list, self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.splited_train_idxs[idx], self.splited_train_weights[idx], num_bin, r1, r2, c1, c2, b)
            
        for idx in range(self.n_test):
            for num_bin, r1, r2, c1, c2, b in zip(self.bins_list, self.ran_r1, self.ran_r2, self.ran_c1, self.ran_c2, self.ran_b):
                k, yk = self.hashing(self.splited_test_idxs[idx], self.splited_test_weights[idx], num_bin, r1, r2, c1, c2, b)
                self.test_samples[idx].append([k, yk])

        self.total_time = '{:.5f}s'.format(time.time() - start)

    def hashing(self, splited_idxs, splited_weights, num_bin, r1, r2, c1, c2, b):
        i, bin_idx = 0, num_bin
        while True:
            idxs = splited_idxs[bin_idx]
            if len(idxs) == 0:
                bin_idx = self.rerandomize_list[num_bin][i]
                i += 1
            else:
                break
        weights = splited_weights[bin_idx]

        minIdx, minVal, minY = 0, sys.maxsize, 0
        for idx, weight in zip(idxs, weights):
            idx %= self.numSigPerBins
            if weight == 0:
                continue
            t = math.floor((math.log(weight) / -math.log(r1[idx] * r2[idx])) + b[idx])
            y = math.exp(-math.log(r1[idx] * r2[idx]) * (t - b[idx]))
            a = -math.log(c1[idx] * c2[idx]) * r1[idx] * r2[idx] / y
            
            if minVal > a:
                minVal = a
                minIdx = idx
                minY = y

        return minIdx, minY
    
    def get_similarity(self):
        self.similarity = []            # total similarity
        self.sorted_closest_idxs = []   # total index sorted with 
        for ts in self.test_samples:
            one_sim = []                # simlarity between one test dataset and rest train dataset
            for trs in self.train_samples:
                cnt = 0
                for t, tr in zip(ts, trs):
                    if t[0] == tr[0] and t[1] == tr[1]:
                        cnt += 1
                one_sim.append(cnt / self.n_sig)

            self.similarity.append(one_sim)
            sorted_similarity_idx = sorted(range(len(one_sim)), key=lambda k: one_sim[k], reverse=True)
            self.sorted_closest_idxs.append(sorted_similarity_idx)
    
    def get_performance(self):
        self.acc = calculate_acc(self.train_labels, self.test_labels, self.sorted_closest_idxs)
        self.prec_list = calculate_prec(self.jacc, self.sorted_closest_idxs, self.kList)

