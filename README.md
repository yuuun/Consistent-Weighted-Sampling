# Improved Consistent Weighted Sampling
As datasets become larger and more high-dimensional, it becomes increasingly important to find data representations that allow compact storage and efficient distance computation and retrieval. Improved Consistent Weighted Sampling(ICWS, IEEE, 20110) is the state-of-the-art of the sampling methods in weighted sets. There are some variants of ICWS which decreases the time of sampling or increase the quality of samples. 

In this repository, I implemented ICWS and its variants.

## Build Instruction
```
$ git clone git@github.com:yuuun/Consistent-Weighted-Sampling.git
$ cd Consistent-Weighted-Sampling
$ pip3 install -r requirements.txt
$ mkdir dataset
```

After Downloading datasets to dataset/ file, you can run the file by the following command
```
$ python3 main.py
```

## Datasets 
 - Datasets can be downloaded in https://www.csie.ntu.edu.tw/~cjlin/libsvm/

### Input File Data Formats
```
[label1] [idx1_1]:[weight1_1] [idx1_2]:[weight1_2] ...
[label2] [idx2_1]:[weight2_1] [idx2_2]:[weight2_2] ...
[label3] [idx3_1]:[weight3_1] [idx3_2]:[weight3_2] ...
```
 
#### Example
```
1 1:5 3:7 10:9
2 2:6 3:1 8:10 
```

## Models
- ICWS
    - state of the CWS method that improves the effectiveness and efficiency
- 0-bit CWS
    - improves the space efficiency and GJS estimation time
- CCWS
    - improves the time efficiency of ICWS by removing the cacluation of logarithm
- PCWS
    - replace one of the two gamma distributions with uniform distribution which improves both time and space complexities in ICWS
- I2CWS
    - hashes $y_k$ and $z_k$ separately which makes $k^{*}$ and $y_{k^*}$ dependent
- SCWS
    - simplifies into a floating-point multiplication using a pool of presented values
    - increases the time efficiency because of the pre-sampled pool
- BCWS
    - applies OPH on CWS and increased the time efficiency dramatically

## Experiments
### Evaluation
 - Hashing Time
 - Classification Accuracy
 - Precision(the criteria of Jaccard Similarity)

### References
 - ICWS: [Improved consistent sampling, weighted minhash and l1 sketching](https://ieeexplore.ieee.org/abstract/document/5693978/?casa_token=cD19RSA8IxUAAAAA:0FWHkkknyJ1pK9Sy9n_saBIeLfS5aajGDw5NBJmPNcfvPShqat8AR5id8Kobp86ZsikbpOoXYrs), Sergey Ioffe, ICDM, 2010
 - 0-bit CWS: [0-bit consistent weighted sampling](https://dl.acm.org/doi/abs/10.1145/2783258.2783406?casa_token=uP0Mu8Z8EDMAAAAA:RYXF3QRGxTbQ7wlEwoNZieO6J5XC2oLHV2cZqDSCX-LUuQpJwDZdy1TSjT_ZzJWTTN7kwjHRyBe94rQ), Ping Li, KDD, 2015
 - CCWS: [Canonical consistent weighted sampling for real-value weighted min-hash](https://ieeexplore.ieee.org/abstract/document/7837987/?casa_token=3TNUkPLz8nYAAAAA:Foee7yZzzhKqUJ67zUehtz-t8GaHoODorolxfAxYWK0aa0KeL7HcB5IVF7wsnC_9oWUrCwdmZck), Wei Wu, NIPS, 2016
 - PCWS: [Consistent weighted sampling mad more practical](https://dl.acm.org/doi/abs/10.1145/3038912.3052598?casa_token=ZucI6adplDYAAAAA:N4rV4dcWQtyhPWS1zZFi4J7IlEdNEQLWN2axJf9sWfW35ylDkTcYI0f1uEHx2tkjfqJJ8AIHgCAU1x0), Wei Wu, WWW, 2016
 - I2CWS: [Improved consistent weighted sampling revisited](https://ieeexplore.ieee.org/abstract/document/8493289/?casa_token=gBsxfXBHNosAAAAA:lLmHk1eYCd0jkBF6-F4A6DsbvZOUAvreLjrTU5BG2ofutdw8cYWHAdMeCmil4kA68ud7TyW-VW4), Wei Wu, WWW, 2017
 - SCWS: [Canonical consistent weighted sampling for real-value weighted min-hash](https://ieeexplore.ieee.org/abstract/document/7837987/?casa_token=yXYVurruAnMAAAAA:5Fq0Gj3ogIBcGll3dLh1xN4c7MuNueS5S9gARGnqNJwYfuUvZHFpBHRi1hK9dxrlxnCVEiJBy6Q), E.Raff, CIKM, 2018
 - BCWS: [Re-randomized densification for one permutation hashing and bin-wise consistent weighted sampling](https://proceedings.neurips.cc/paper/2019/hash/9f067d8d6df2d4b8c64fb4c084d6c208-Abstract.html), Ping Li, NIPS, 2019
