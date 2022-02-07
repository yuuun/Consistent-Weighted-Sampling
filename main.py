from data_loader import Data
from models import *
import pickle

if __name__=='__main__':
    data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB', 'mnist']
    data_id = -1

    ### Since calculating Jaccard Similarity taked much time, we save the results of datasets and Jaccard Simliarity
    data = Data(data_id)
    with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'wb') as f:
        pickle.dump(data, f)

    js = GeneralizedJaccardSimliarity(data.train_weights, data.train_idxs, data.test_weights, data.test_idxs)
    js.calculate_jacc()
    '''

    with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'rb') as f:
        data = pickle.load(f)
    
    with open('./dataset/{0}.js'.format(data_list[data_id]), 'rb') as f:
        js = pickle.load(f)
    '''