## Data
1. [iac.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac.csv) is the benchmark which contains a column for the ground truth template location and a column for the prompt.
2. [process_iac.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/process_iac.csv) is also the benchmark, but with additional information about each row.
3. [iac_user_intent.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac_user_intent.csv) contains the necessary information for the user intent validation.


### process_iac.csv
- process_iac.csv is generated after running the [process_dataset.py](https://github.com/Tianyi2/IaCGen/blob/main/Data/process_dataset.py) which takes the iac.csv as input.
- process_iac.csv include additional information:
  - `difficulty_level`: (1~5) difficulty level as discussed in the paper.
  - `resources`: All AWS resource names included in the template.
  - `resource_count`: Number of AWS resources included in the template.
  - `loc`: Line of code included in the template (whitespace will not be counted).
  - `parameter_count`: Number of parameters included in the template.


### Difficulty Levels
| **Difficulty level** | **LOC**   | **Resource** | **Parameter** |
|----------------------|-----------|--------------|----------------|
| 1                    | < 50      | < 2          | < 2            |
| 2                    | < 100     | < 4          | < 5            |
| 3                    | < 150     | < 6          | < 9            |
| 4                    | < 200     | < 12         | < 14           |
| 5                    | ≥ 200     | ≥ 12         | ≥ 14           |
