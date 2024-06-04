# This is the code of the H2KT model. Below is a list of the main execution steps of the model training, taking the JunYi dataset as an example.

### 1. Open the link `https://pslcdatashop.web.cmu.edu/Files?datasetId=1198`, download `junyi_Exercise_table.csv` and `junyi_ProblemLog_original.zip` in the JunYi dataset and decompress them to the current directory.

### 2. After executing `1dataTransfer.py`, the junyi.csv file will be generated in the current directory.

### 3. Execute the `2junyi_data.py` file to generate relevant data in six lines.

### 4. Execute `code_all/train.py` to train and test the H2KT model.