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


