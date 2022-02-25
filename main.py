from models import *
from utils import *

if __name__=='__main__':
    data_list = ['satimage.scale', 'Sensorless', 'usps', 'scene_', 'cifar10', 'SVHN', 'gisette_scale', 'smallNORB', 'mnist']
    dim_list = [36, 48, 256, 294, 3072, 3072, 5000, 18432, 780]
    data_id = -1
    dim = dim_list[data_id]

    data_path = './dataset/{0}'.format(data_list[data_id])
    
    # if you don't need to save the result of each model, you can use the parameter of do_save=False
    data = load_data(data_id, data_path + '.pkl')
   
    js = load_js(data, data_path + '.js', GeneralizedJaccardSimliarity)
    
    ### We figured out that the dimension does not exactly match between the total dataset and train dataset.
    ### So we got dimensions from each datasets depcited in LIBSVM
    
    n_sig = 100
    kList = [1, 10, 50, 100, 500]

    icws = load_model(data, data_path + '.icws', ICWS, dim, n_sig, kList, js)
    zero_icws = load_model(data, data_path + '.zero_icws', zero_ICWS, dim, n_sig, kList, js)
    ccws = load_model(data, data_path + '.ccws', CCWS, dim, n_sig, kList, js)
    pcws = load_model(data, data_path + '.pcws', PCWS, dim, n_sig, kList, js)
    i2cws = load_model(data, data_path + '.i2cws', I2CWS, dim, n_sig, kList, js)
    bcws = load_model(data, data_path + '.bcws', BCWS, dim, n_sig, kList, js)
    import pdb; pdb.set_trace()