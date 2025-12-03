## How to use
- install dependency for main.py
  - using cu118 for pytorch will ensure same results for `pred.csv`
  - but if the environment is not compatible, simply install all the dependencies
    ```
    pip install torch
    pip install pandas
    pip install numpy
    pip install matplotlib
    pip install scikit-learn
    pip install seaborn
    ```
- place `train.csv`, `test4students.csv` under the root folder
  - must under the same directory as `improved_cnn.sh`, `main.py`
- execute `improved_cnn.sh`
  ```
  bash improved_cnn.sh cuda
  ```
  or 
  ```
  bash improved_cnn.sh
  ```
- you will see the results under folder `results_cnn_improved/bn1_do0_res0_poolavg/`
  ```
  results_cnn_improved/bn1_do0_res0_pool_avg/
  ├── best_model.pth
  ├── confusion_matrix.png
  ├── learning_curve.png
  ├── log.txt
  └── pred.csv
  ```