import os
import pickle
from data_loader import Data

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


def load_data(data_id, data_path):
    ### Since calculating Jaccard Similarity taked much time, we save the results of datasets and Jaccard Simliarity
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = Data(data_id)
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    print('finishing loading data')
    return data

def load_js(data, data_path, Model):
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        js = Model(data.train_weights, data.train_idxs, data.train_labels, data.test_weights, data.test_idxs, data.test_labels)
        with open(data_path, 'wb') as f:
            pickle.dump(js, f)

    print('finishing loading ' + js)
    return js

def load_model(data, data_path, Model, dim, n_sig, kList, js):
    data_path = data_path + '_' + str(n_sig)
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            model =  pickle.load(f)
    else:
        model = Model(data.train_weights, data.train_idxs, data.train_labels,
                            data.test_weights, data.test_idxs, data.test_labels,
                            js.sorted_closest_idxs, dim, n_sig, kList)
        with open(data_path, 'wb') as f:
            return pickle.dump(model, f)

    print('finishing loading ' + data_path)
    return model