from data_loader import Data
from models import *
import pickle
from utils import *

if __name__=='__main__':
    data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB', 'mnist']
    data_id = -1

    ### Since calculating Jaccard Similarity taked much time, we save the results of datasets and Jaccard Simliarity
    
    # data = Data(data_id)
    # with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'wb') as f:
    #     pickle.dump(data, f)

    with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'rb') as f:
        data = pickle.load(f)

    # js = GeneralizedJaccardSimliarity(data.train_weights, data.train_idxs, data.train_labels, data.test_weights, data.test_idxs, data.test_labels)
    # with open('./dataset/{0}.js'.format(data_list[data_id]), 'wb') as f:
    #     pickle.dump(js, f)

    with open('./dataset/{0}.js'.format(data_list[data_id]), 'rb') as f:
        js = pickle.load(f)
    
    ### We figured out that the dimension does not exactly match between the total dataset and train dataset.
    ### So we got dimensions from each datasets depcited in LIBSVM
    dim_list = [36, 48, 256, 294, 3072, 3072, 5000, 18432, 780]
    n_sig = 200
    kList = [1, 10, 50, 100, 500]

    # icws = ICWS(data.train_weights, data.train_idxs, data.train_labels,
    #             data.test_weights, data.test_idxs, data.test_labels,
    #             js.sorted_closest_idxs, dim_list[data_id], n_sig, kList)
    # with open('./dataset/{0}.icws'.format(data_list[data_id]), 'wb') as f:
    #     pickle.dump(icws, f)
    
    with open('./dataset/{0}.icws'.format(data_list[data_id]), 'rb') as f:
        icws = pickle.load(f)
    
    zero_icws = zero_ICWS(data.train_weights, data.train_idxs, data.train_labels,
                            data.test_weights, data.test_idxs, data.test_labels,
                            js.sorted_closest_idxs, dim_list[data_id], n_sig, kList)
    
    with open('./dataset/{0}.zero_icws'.format(data_list[data_id]), 'wb') as f:
        pickle.dump(zero_icws, f)
    
    with open('./dataset/{0}.zero_icws'.format(data_list[data_id]), 'rb') as f:
        zero_icws = pickle.load(f)

    pcws = PCWS(data.train_weights, data.train_idxs, data.train_labels,
                            data.test_weights, data.test_idxs, data.test_labels,
                            js.sorted_closest_idxs, dim_list[data_id], n_sig, kList)
    
    with open('./dataset/{0}.pcws'.format(data_list[data_id]), 'wb') as f:
        pickle.dump(pcws, f)
    
    with open('./dataset/{0}.pcws'.format(data_list[data_id]), 'rb') as f:
        pcws = pickle.load(f)