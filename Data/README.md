## Data
1. [iac.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac.csv) is the benchmark which contains a column for the ground truth template location and a column for the prompt.
2. [process_iac.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/process_iac.csv) is also the benchmark with additional information about each row, such as difficulty level, line of code, number of parameters, number of resources, types of resources.
3. process_iac.csv is generated after running the process_dataset.py which takes the iac.csv as input then 