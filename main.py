from data_loader import Data
from models import *
import pickle

if __name__=='__main__':
    data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB', 'mnist']
    data_id = -1

    ### Since calculating Jaccard Similarity taked much time, we save the results of datasets and Jaccard Simliarity
    '''
    data = Data(data_id)
    with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'wb') as f:
        pickle.dump(data, f)

    js = GeneralizedJaccardSimliarity(data.train_weights, data.train_idxs, data.test_weights, data.test_idxs)
    js.calculate_jacc()
    with open('./dataset/{0}.js'.format(data_list[data_id]), 'wb') as f:
        pickle.dump(js, f)
    '''

    with open('./dataset/{0}.pkl'.format(data_list[data_id]), 'rb') as f:
        data = pickle.load(f)
    
    with open('./dataset/{0}.js'.format(data_list[data_id]), 'rb') as f:
        js = pickle.load(f)
    
    ### We figured out that the dimension does not exactly match between the total dataset and train dataset.
    ### So we got dimensions from each datasets depcited in LIBSVM
    dim_list = [36, 48, 256, 294, 3072, 3072, 5000, 18432, 780]
    # icws = ICWS(data.train_weights, data.train_idxs, 
    #             data.test_weights, data.test_idxs, 
    #             js.sorted_closest_idxs, dim_list[data_id], 50)

    # with open('./dataset/{0}.icws'.format(data_list[data_id]), 'wb') as f:
    #     pickle.dump(icws, f)
    
    with open('./dataset/{0}.icws'.format(data_list[data_id]), 'rb') as f:
        icws = pickle.load(f)
    cnt = 0
    for sigs in icws.train_samples:
        if sigs[0][0] == icws.test_samples[0][0][0] and sigs[0][1] == icws.test_samples[0][0][1]:
            cnt += 1
    
    simiarity = []
    for ts in icws.test_samples:
        sim = []
        for trs in icws.train_samples:
            cnt = 0
            for t, tr in zip(ts, trs):
                if t[0] == tr[0] and t[1] == tr[1]:
                    cnt += 1
            sim.append(cnt / 50)
        simiarity.append(sim)
                    

    import pdb; pdb.set_trace()