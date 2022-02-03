from winreg import DeleteKey


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

    def load_data(self, data_path):
        
        return