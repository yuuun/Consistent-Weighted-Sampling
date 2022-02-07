import time
class GeneralizedJaccardSimliarity():
    def __init__(self, train_weights, train_idxs, test_weights, test_idxs):
        self.train_weights = train_weights
        self.train_idxs = train_idxs
        self.test_weights = test_weights
        self.test_idxs = test_idxs
    
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

            sorted_similarity_idx = sorted(range(len(similarity)), key=lambda k: similarity[k])
            self.sorted_closest_idxs.append(sorted_similarity_idx)
            self.total_time = '{:.5f}s'.format(time.time() - start)