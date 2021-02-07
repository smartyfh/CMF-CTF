## CMF

This repo maintains the implementation of the Cauchy matrix factorization method for Web service QoS prediction.

### Usage

1. Download dataset

   The adopted dataset can be downloaded from [WSDREAM](https://github.com/wsdream/wsdream-dataset).

2. Remove outliers from the dataset, e.g.,

   ```
     $ python3 Full_outlier.py 0.1 "rt"
   ```
   
3. Train the model, e.g.,

   ```
     $ python3 CMF_qos.py 0.5 0.1 "rt"
   ```
   
   The parameters in the main function should be changed as per the request.
