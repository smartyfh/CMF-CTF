## CTF

This repo maintains the implementation of the Cauchy tensor factorization method for time-aware Web service QoS prediction.

### Usage

1. Download dataset

   The adopted dataset can be downloaded from [WSDREAM](https://github.com/wsdream/wsdream-dataset).

2. Remove outliers from the dataset, e.g.,

   ```
   python3 Full_time_outlier.py 0.1 "rt"
   ```
   
3. Train the model

   * CTF is implemented in Cython, so we need to compile the code at first
   
      ```
      python3 setup.py build_ext --inplace
      ```
      
      Line 25 in setup.py need to be modified in accordance with the running environment.
      
   * Then, train the model
   
      ```
      python3 run.py
      ```
   
      The configurations can be changed in run.py.
