## Data
1. [iac_basic.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac_basic.csv) is the benchmark with basic information of two columns for the ground truth template location and the prompt.
2. [iac_with_deployability.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac_with_deployability.csv) is the benchmark with addtional information about the feedback used when execute IaCGen.
3. [iac_with_difficulty_levels.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac_with_difficulty_levels.csv) is the benchmark with the difficulty level and additional info about each row.
4. [iac_with_user_intent.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/iac_with_user_intent.csv) is the benchmark with necessary information for the user intent validation.
5. [iac_with_user_intent.csv](https://github.com/Tianyi2/IaCGen/blob/main/Data/benchmark_included_aws_resources.csv) present all the AWS services included in the benchmark.


### iac_with_difficulty_levels.csv
- iac_with_difficulty_levels.csv is generated after running the [process_dataset.py](https://github.com/Tianyi2/IaCGen/blob/main/Data/process_dataset.py) which takes the iac_basic.csv as input.
- iac_with_difficulty_levels.csv include additional information:
  - `difficulty_level`: (1~5) difficulty level as discussed in the paper.
  - `resources`: All AWS resource names included in the template.
  - `resource_count`: Number of AWS resources included in the template.
  - `loc`: Line of code included in the template (whitespace will not be counted).
  - `parameter_count`: Number of parameters included in the template.


### iac_with_user_intent.csv
- iac_with_user_intent.csv is built upon the iac_with_difficulty_levels.csv file by adding the information for user intent validation:
  - `user_intent_file_path`: List of user intent specification file pathes that will be used to validate against the LLM-generated IaC.
  - `user_intent_id`: Id of user intent specification files.
  - `needed_resources`: List of needed AWS resources for the IaC template.  


### Difficulty Levels
| **Difficulty level** | **LOC**   | **Resource** | **Parameter** |
|----------------------|-----------|--------------|----------------|
| 1                    | < 50      | < 2          | < 2            |
| 2                    | < 100     | < 4          | < 5            |
| 3                    | < 150     | < 6          | < 9            |
| 4                    | < 200     | < 12         | < 14           |
| 5                    | ≥ 200     | ≥ 12         | ≥ 14           |
