# Improved Consistent Weighted Sampling
This is the code of ICWS and its variants.

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
[label1] [idx1_1]:[val1_1] [idx1_2]:[val1_2] ...
[label2] [idx2_1]:[val2_1] [idx2_2]:[val2_2] ...
[label3] [idx3_1]:[val3_1] [idx3_2]:[val3_2] ...
```
 
#### Example
```
1 1:5 3:7 10:9
2 2:6 3:1 8:10 
```

## Models
- ICWS
- 0-bit CWS
- CCWS
- PCWS
- I2CWS
- SCWS
- BCWS

## Experiments
### Evaluation
 - Hashing Time
 - Classification Accuracy
 - Precision(the criteria of Jaccard Similarity)

### References
 - ICWS: [Improved consistent sampling, weighted minhash and l1 sketching](https://ieeexplore.ieee.org/abstract/document/5693978/?casa_token=cD19RSA8IxUAAAAA:0FWHkkknyJ1pK9Sy9n_saBIeLfS5aajGDw5NBJmPNcfvPShqat8AR5id8Kobp86ZsikbpOoXYrs)
 - 0-bit CWS: [0-bit consistent weighted sampling](https://dl.acm.org/doi/abs/10.1145/2783258.2783406?casa_token=uP0Mu8Z8EDMAAAAA:RYXF3QRGxTbQ7wlEwoNZieO6J5XC2oLHV2cZqDSCX-LUuQpJwDZdy1TSjT_ZzJWTTN7kwjHRyBe94rQ)
 - CCWS: [Canonical consistent weighted sampling for real-value weighted min-hash](https://ieeexplore.ieee.org/abstract/document/7837987/?casa_token=3TNUkPLz8nYAAAAA:Foee7yZzzhKqUJ67zUehtz-t8GaHoODorolxfAxYWK0aa0KeL7HcB5IVF7wsnC_9oWUrCwdmZck)
 - PCWS: [Consistent weighted sampling mad more practical](https://dl.acm.org/doi/abs/10.1145/3038912.3052598?casa_token=ZucI6adplDYAAAAA:N4rV4dcWQtyhPWS1zZFi4J7IlEdNEQLWN2axJf9sWfW35ylDkTcYI0f1uEHx2tkjfqJJ8AIHgCAU1x0)
 - I2CWS: [Improved consistent weighted sampling revisited](https://ieeexplore.ieee.org/abstract/document/8493289/?casa_token=gBsxfXBHNosAAAAA:lLmHk1eYCd0jkBF6-F4A6DsbvZOUAvreLjrTU5BG2ofutdw8cYWHAdMeCmil4kA68ud7TyW-VW4)
 - SCWS: [Canonical consistent weighted sampling for real-value weighted min-hash](https://ieeexplore.ieee.org/abstract/document/7837987/?casa_token=yXYVurruAnMAAAAA:5Fq0Gj3ogIBcGll3dLh1xN4c7MuNueS5S9gARGnqNJwYfuUvZHFpBHRi1hK9dxrlxnCVEiJBy6Q)
 - BCWS: [Re-randomized densification for one permutation hashing and bin-wise consistent weighted sampling](https://proceedings.neurips.cc/paper/2019/hash/9f067d8d6df2d4b8c64fb4c084d6c208-Abstract.html)
