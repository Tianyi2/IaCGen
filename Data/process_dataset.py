import pandas as pd
import yaml
import json
import os
import sys
from Code.evaluation.cloud_evaluation import get_required_resource_types


def count_parameters(template_path):
    """Count parameters in template file"""
    # Determine file type from extension
    is_json = template_path.lower().endswith('.json')

    # Custom YAML loader for CloudFormation
    class CloudFormationLoader(yaml.SafeLoader):
        pass

    # Add constructors for CloudFormation intrinsic functions
    def construct_cfn_tag(loader, node):
        if isinstance(node, yaml.ScalarNode):
            return node.value
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)

    # Register all common CloudFormation tags
    cfn_tags = ['!Ref', '!Sub', '!GetAtt', '!Join', '!Select', '!Split', '!Equals', '!If',
                '!FindInMap', '!GetAZs', '!Base64', '!Cidr', '!Transform', '!ImportValue',
                '!Not', '!And', '!Or', '!Condition', '!ForEach', '!ValueOf', '!Rain::Embed']
    for tag in cfn_tags:
        CloudFormationLoader.add_constructor(tag, construct_cfn_tag)

    with open(template_path, 'r') as f:
        if is_json:
            template = json.load(f)
        else:
            template = yaml.load(f, Loader=CloudFormationLoader)

    parameters = template.get('Parameters', {})

    return len(parameters)


def count_lines(template_path):
    """Count lines of code in template file"""
    try:
        with open(template_path, 'r') as file:
            return sum(1 for line in file if line.strip())
    except Exception as e:
        print(f"Error counting lines in {template_path}: {str(e)}")
        return 0


def calculate_difficulty(loc, resource_count, param_count):
    """Calculate difficulty level based on LOC, resource count, and parameter count"""
    if loc < 50 and resource_count < 2 and param_count < 2:
        return 1
    elif loc < 100 and resource_count < 4 and param_count < 5:
        return 2
    elif loc < 150 and resource_count < 8 and param_count < 9:
        return 3
    elif loc < 200 and resource_count < 12 and param_count < 14:
        return 4
    else:
        return 5


def start_process(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv, encoding='latin-1')

    # Add new columns
    df['difficulty_level'] = 0
    df['resources'] = ''
    df['resource_count'] = 0
    df['loc'] = 0
    df['parameter_count'] = 0

    # Process each row
    for index, row in df.iterrows():
        ground_truth_path = str(row['ground_truth_path'])
        if os.path.exists(ground_truth_path):
            # Count metrics
            loc = count_lines(ground_truth_path)
            param_count = count_parameters(ground_truth_path)

            # Update DataFrame
            df.at[index, 'loc'] = loc
            types = get_required_resource_types(ground_truth_path)
            df.at[index, 'resources'] = ', '.join(types['resource_types'])
            df.at[index, 'resource_count'] = types['total_resources']
            df.at[index, 'parameter_count'] = param_count
            df.at[index, 'difficulty_level'] = calculate_difficulty(loc, types['total_resources'], param_count)

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Process completed. Results saved to {output_csv}")


if __name__ == "__main__":
    print("Start Process")
    input_csv = "iac.csv"
    output_csv = "process_iac.csv"
    start_process(input_csv, output_csv)
    print("End Process")
