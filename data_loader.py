import random as rd
import sys

class Data():
    def __init__(self, data_id):
        data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB']
        train_format_list = ['.tr', '', '', 'train', '', '', '', '']
        test_format_list = ['.t', '', '.t', 'test', '.t', '.t', '.t', '.t']
        file_nm = data_list[data_id]

        self.set_num(file_nm + train_format_list[data_id], file_nm + test_format_list[data_id])
    
    def set_num(self, data_path_train, data_path_test):
        self.numOfTest, self.numOfTrain = 1000, 5000
        lines = open(data_path_train, 'r').readlines()
        num_train = len(lines)
        
        if data_path_test == data_path_train:
            if num_train < self.numOfTest + self.numOfTrain:
                self.numOfTrain = (int)(num_train / 2.) - 1
                self.numOfTest = (int)(num_train / 2.) - 1
                
        else:
            if num_train < self.numOfTrain:
                self.numOfTrain = num_train
        
        self.total_train = num_train
        self.total_test = len(open(data_path_test, 'r').readlines())

    def load_data(self, data_path_train, data_path_test):
        if data_path_train == data_path_test:
            weights, labels = self.load_weights(data_path_train)
        else:
            train_weights, train_labels = self.load_weights(data_path_train)
            test_weights, test_labels = self.load_weights(data_path_test)

        
        return
    
    def load_weights(self, data_path):
        lines = open(data_path, 'r').readlines()
        weights = []

        labels = []
        for l in lines:
            tmp = l.split(' ')
            labels.append(tmp[0])
            sub_weight = {}
            for tm in tmp:
                t = tm.split(':')
                sub_weight[t[0]] = t[1]
            weights.append(sub_weight)

        return weights, labels
    
    def scale_weight(self):
        min_val, max_val = float('inf'), float('-inf')
    
        for tw in self.train_weights:
            values = list(tw.values())
            min_val = min(min_val, min(values))
            max_val = max(max_val, max(values))
        
