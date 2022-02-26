# Improved Consistent Weighted Sampling
As datasets become larger and more high-dimensional, it becomes increasingly important to find data representations that allow compact storage and efficient distance computation and retrieval. Improved Consistent Weighted Sampling(ICWS, IEEE, 2010) is the state-of-the-art of the sampling methods in weighted sets. There are some variants of ICWS which decreases the time of sampling or increase the quality of samples. 

This repository implements ICWS and its variants.

## Build Instruction
```
$ git clone git@github.com:yuuun/Consistent-Weighted-Sampling.git
$ cd Consistent-Weighted-Sampling
$ pip3 install -r requirements.txt
$ mkdir dataset
```

After Downloading datasets to 'dataset/' file, you can run the file by the following command
```
$ python3 main.py
```

## Datasets 
 - Datasets can be downloaded in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

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
    - hashes y<sub>k</sub> and z<sub>k</sub> separately which makes k<sup>'</sup> and y<sub>k<sup>'</sup></sub> dependent
- BCWS
    - applies OPH on CWS and increased the time efficiency dramatically

## Experiments
#### The classification accuracy of Generalized Jaccard Similarity
| |Classification Accuracy|
|---|---|
|Mnist|93.6%|
### Evaluation
 - Hashing Time
 - Classification Accuracy
 - Precision(the criteria of Jaccard Similarity)
 
 - The experiments will be depicted as follows,
 
 [Name of Method] | [Dataset]([Number of Samples])

 #### Time
 |Method|Mnist(100)|Mnist(500)|
 |-----|------|--|
 |ICWS|175.4s|869.6s|
 |0-bit CWS|169.3s|842.1s|
 |CCWS|129.2s|647.4s|
 |PCWS|154.4s|767.0s|
 |I2CWS|162.3s|805.7s|
 |BCWS|6.3s|17.9s|

 #### Accuracy
 |Method|Mnist(100)|Mnist(500)|
 |-----|------|--|
 |ICWS|88.7%|92.6%|
 |0-bit CWS|90.7%|94.1%|
 |CCWS|79.3%|81.8%|
 |PCWS|90.6%|93.2%|
 |I2CWS|89.7%|92.5%|
 |BCWS|84.3%|84.4%|

 #### Precision@10
 |Method|Mnist(100)|Mnist(500)|
 |-----|------|--|
 |ICWS|59.9%|80.4%|
 |0-bit CWS|61.6%|80.5%|
 |CCWS|35.43%|40.8%|
 |PCWS|60.9%|79.5%|
 |I2CWS|59.2%|78.8%|
 |BCWS|46.4%|46.4%|

 #### Precision@100
|Method|Mnist(100)|Mnist(500)|
 |-----|------|---|
 |ICWS|71.1%|86.8%|
 |0-bit CWS|71.8%|86.3%|
 |CCWS|48.7%|54.8%|
 |PCWS|70.5%|85.8%|
 |I2CWS|69.9%|84.8%|
 |BCWS|57.5%|52.8%|
 


### References
 - ICWS: [Improved consistent sampling, weighted minhash and l1 sketching](https://ieeexplore.ieee.org/abstract/document/5693978/?casa_token=cD19RSA8IxUAAAAA:0FWHkkknyJ1pK9Sy9n_saBIeLfS5aajGDw5NBJmPNcfvPShqat8AR5id8Kobp86ZsikbpOoXYrs), Sergey Ioffe, ICDM, 2010
 - 0-bit CWS: [0-bit consistent weighted sampling](https://dl.acm.org/doi/abs/10.1145/2783258.2783406?casa_token=uP0Mu8Z8EDMAAAAA:RYXF3QRGxTbQ7wlEwoNZieO6J5XC2oLHV2cZqDSCX-LUuQpJwDZdy1TSjT_ZzJWTTN7kwjHRyBe94rQ), Ping Li, KDD, 2015
 - CCWS: [Canonical consistent weighted sampling for real-value weighted min-hash](https://ieeexplore.ieee.org/abstract/document/7837987/?casa_token=3TNUkPLz8nYAAAAA:Foee7yZzzhKqUJ67zUehtz-t8GaHoODorolxfAxYWK0aa0KeL7HcB5IVF7wsnC_9oWUrCwdmZck), Wei Wu, NIPS, 2016
 - PCWS: [Consistent weighted sampling mad more practical](https://dl.acm.org/doi/abs/10.1145/3038912.3052598?casa_token=ZucI6adplDYAAAAA:N4rV4dcWQtyhPWS1zZFi4J7IlEdNEQLWN2axJf9sWfW35ylDkTcYI0f1uEHx2tkjfqJJ8AIHgCAU1x0), Wei Wu, WWW, 2016
 - I2CWS: [Improved consistent weighted sampling revisited](https://ieeexplore.ieee.org/abstract/document/8493289/?casa_token=gBsxfXBHNosAAAAA:lLmHk1eYCd0jkBF6-F4A6DsbvZOUAvreLjrTU5BG2ofutdw8cYWHAdMeCmil4kA68ud7TyW-VW4), Wei Wu, WWW, 2017
 - BCWS: [Re-randomized densification for one permutation hashing and bin-wise consistent weighted sampling](https://proceedings.neurips.cc/paper/2019/hash/9f067d8d6df2d4b8c64fb4c084d6c208-Abstract.html), Ping Li, NIPS, 2019
