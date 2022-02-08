import random as rd

class Data():
    def __init__(self, data_id):
        data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB', 'mnist']
        train_format_list = ['.tr', '', '', 'train', '', '', '', '', '']
        test_format_list = ['.t', '', '.t', 'test', '.t', '.t', '.t', '.t', '.t']
        
        file_nm = './dataset/' + data_list[data_id]
        data_path_train = file_nm + train_format_list[data_id]
        data_path_test = file_nm + test_format_list[data_id]

        self.set_num(data_path_train, data_path_test)
        self.load_data(data_path_train, data_path_test)
    
    def set_num(self, data_path_train, data_path_test):
        '''
        Since we set the number of train dataset and test dataset 5000 and 1000 respectively, 
        there are some datasets that has less than those given sets.

        Here, we count the total number of train and test dataset given in LIBSVM.
        If the number of train datasets given in LIBSVM is smaller than the number we set, we set those numbers as train and test datasets.
        
        Some of the datasets have only train datasets, excluding test datasets.
        '''

        self.numOfTest, self.numOfTrain = 1000, 5000
        lines = open(data_path_train, 'r').readlines()
        num_train = len(lines)
        
        if data_path_test == data_path_train:
            if num_train < self.numOfTest + self.numOfTrain:
                if num_train < self.numOfTest:
                    self.numOfTrain = (int)(num_train / 2.) - 1
                    self.numOfTest = (int)(num_train / 2.) - 1
                else:
                    self.numOfTrain = num_train - self.numOfTest
        else:
            if num_train < self.numOfTrain:
                self.numOfTrain = num_train
            self.total_test = len(open(data_path_test, 'r').readlines())
        
        self.total_train = num_train

    def load_data(self, data_path_train, data_path_test):
        if data_path_train == data_path_test:
            idxs_list, weights_list, labels = self.load_weights(data_path_train)
            train_test_idxs = rd.sample(list(range(self.total_train)), self.numOfTrain, self.numOfTest)
            
            tr_idxs = train_test_idxs[:self.numOfTrain]
            t_idxs = train_test_idxs[self.numOfTrain:]

            self.train_idxs, self.train_weights, self.train_labels = [], [], []
            for idxs in tr_idxs:
                self.train_idxs.append(idxs_list[idxs])
                self.train_weights.append(weights_list[idxs])
                self.train_labels.append(labels[idxs])

            self.test_idxs, self.test_weights, self.test_labels = [], [], []
            for idxs in t_idxs:
                self.test_idxs.append(idxs_list[idxs])
                self.test_weights.append(weights_list[idxs])
                self.test_labels.append(labels[idxs])
            
        else:
            idxs_list, weights_list, labels = self.load_weights(data_path_train)
            tr_idxs = rd.sample(list(range(self.total_train)), self.numOfTrain)
            self.train_idxs, self.train_weights, self.train_labels = [], [], []
            for idxs in tr_idxs:
                self.train_idxs.append(idxs_list[idxs])
                self.train_weights.append(weights_list[idxs])
                self.train_labels.append(labels[idxs])
            
            idxs_list, weights_list, labels = self.load_weights(data_path_test)
            t_idxs = rd.sample(list(range(self.total_test)), self.numOfTest)
            self.test_idxs, self.test_weights, self.test_labels = [], [], []
            for idxs in t_idxs:
                self.test_idxs.append(idxs_list[idxs])
                self.test_weights.append(weights_list[idxs])
                self.test_labels.append(labels[idxs])
            
            self.scale_weight()
    
    def load_weights(self, data_path):
        '''
        The row of datasets are consisted as follow,
        [label number] [idx1]:[weight] [idx2]:[weight] .....
        '''
        lines = open(data_path, 'r').readlines()

        idxs_list, weights_list = [], []

        labels = []
        for l in lines:
            tmp = l.split(' ')
            labels.append(int(tmp[0]))
            idxs, weights = [], []
            for tm in tmp[1:]:
                t = tm.split(':')
                idxs.append(int(t[0]))
                weights.append(int(t[1]))
            
            idxs_list.append(idxs)
            weights_list.append(weights)

        return idxs_list, weights_list, labels
    
    def scale_weight(self, alpha=0):
        '''
        Scaling the weight in the scale of [0, 1]
        After scaling, some weights turned into zero which is meaningless.
        For those datasets, we scaled the weights in the range of [alpha, 1]
        '''
        min_val, max_val = float('inf'), float('-inf')
    
        for values in self.train_weights:
            min_val = min(min_val, min(values))
            max_val = max(max_val, max(values))
        
        for i, tw in enumerate(self.train_weights):
            for j in range(len(tw)):
                self.train_weights[i][j] = (self.train_weights[i][j] - min_val) * (1 - alpha) / (max_val - min_val + 1e-9) + alpha

        for i, tw in enumerate(self.test_weights):
            for j in range(len(tw)):
                self.test_weights[i][j] = (self.test_weights[i][j] - min_val) * (1 - alpha) / (max_val - min_val + 1e-9) + alpha

