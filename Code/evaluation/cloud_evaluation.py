import subprocess
import json
import yaml
import boto3
from botocore.exceptions import ClientError
import time
import uuid
import pandas as pd
import datetime
from yamllint import linter
from yamllint.config import YamlLintConfig


# Not using, do not support yaml file currently. 
def count_resources(template_path):
    """ 
    Count the number of resources in a template.
    Note: the template need to be in JSON form.
    """
    # Determine file type from extension
    # is_json = template_path.lower().endswith('.json')
    
    # with open(template_path, 'r') as f:
    #     if is_json:
    #         template = json.load(f)
    #     else:
    #         template = yaml.safe_load(f)

    with open(template_path, 'r') as f:
        template = json.load(f)
    
    resources = template.get('Resources', {})
    
    metrics = {
        'total_resources': len(resources),
        'resource_types': len(set(r['Type'] for r in resources.values())),
        'iam_resources': len([r for r in resources.values() if 'AWS::IAM::' in r['Type']]),
        'security_groups': len([r for r in resources.values() if r['Type'] == 'AWS::EC2::SecurityGroup'])
    }
    
    return metrics


# Helper for analyze_resource_coverage
def get_required_resource_types(template_path):
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
    
    resources = template.get('Resources', {})
    
    types = {
        'total_resources': len(resources),
        'resource_types': [r['Type'] for r in resources.values()],
    }

    return types


# Useful - Yaml Syntax Correctness
def yaml_syntax_validation(template_path):
    """
    Validate YAML syntax using yamllint.
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    # Define custom yamllint configuration
    yaml_config = YamlLintConfig('''
        extends: default
        rules:
            document-start: disable
            line-length: disable
            trailing-spaces: disable
            new-line-at-end-of-file: disable
            indentation:
                spaces: consistent
                indent-sequences: consistent
            truthy:
                allowed-values: ['true', 'false', 'yes', 'no']
    ''')

    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Run yamllint
        problems = list(linter.run(template_content, yaml_config))
        
        if problems:
            # Collect all errors and warnings
            error_messages = []
            for problem in problems:
                if problem.level == "error":
                    error_messages.append(f"Line {problem.line}: {problem.desc}")
            
            if error_messages:
                return False, "\n".join(error_messages)
        
        return True, None

    except Exception as e:
        return False, str(e)
    

# Useful - Template Syntax Correctness
def evaluate_template_with_linter(template_path):
    """
    Evalulate the CloudFormation template with AWS CloudFormation Linter.
    Return:
        passed: Boolean -> If the template passes the validation
        total_issues: Integer -> The number of issues find by linter.
        severity_breakdown: Dict -> The number of informations, warnings, and errors find by linter.
    """
    # Run cfn-lint with JSON output
    result = subprocess.run(['cfn-lint', '-f', 'json', template_path], 
                          capture_output=True, text=True)
    
    # Parse the output
    if result.stdout:
        errors = json.loads(result.stdout)
    else:
        errors = []

    # Calculate metrics
    validation_passed = len(errors) == 0
    error_count = len(errors)
    
    # Categorize errors by severity
    error_by_severity = {
        'informational': len([e for e in errors if e['Level'] == 'Informational']),
        'warning': len([e for e in errors if e['Level'] == 'Warning']),
        'error': len([e for e in errors if e['Level'] == 'Error'])
    }

    error_details = []
    # Get error detail when fail syntax validation
    if not validation_passed:   
        for error in errors:
            error_info = {
                'resource': (error.get('Location', {}).get('Path', [])[1] 
                           if error.get('Location', {}).get('Path') is not None 
                           and len(error.get('Location', {}).get('Path', [])) > 1 
                           else None),
                'message': error.get('Message', ''),
                'line_number': error.get('Location', {}).get('Start', {}).get('LineNumber'),
                'rule_description': error.get('Rule', {}).get('Description', ''),
                'documentation': error.get('Rule', {}).get('Source', '')
            }
            error_details.append(error_info)

    return {
        'passed': validation_passed,
        'total_issues': error_count,
        'severity_breakdown': error_by_severity,
        'error_details': error_details
    }


# Useful - Deploy Correctness
def evaluate_template_deployment(template_path):
    """
    Evaluate if a CloudFormation template can be successfully deployed to AWS.
    
    Returns:
        dict containing:
            - success (bool): Whether the deployment was successful
            - error_message (str): Error message if deployment failed
            - stack_id (str): ID of the created stack if successful
            - failed_reason (list): List of reason that why resources failed to create
            - completed_resources (list): List of resources that were created successfully
            - stack_events (list): Detailed stack events if deployment failed
    """
    # Initialize CloudFormation client
    cfn_client = boto3.client('cloudformation')
    
    # Read template content
    with open(template_path, 'r') as f:
        template_body = f.read()
    
    try:
        # Generate a unique stack name
        stack_name = f'validation-stack-{uuid.uuid4().hex[:8]}'
        
        # Create the stack
        create_response = cfn_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            OnFailure='DELETE',  # Automatically delete the stack if it fails
            Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']  # Add necessary capabilities
        )
        
        stack_id = create_response['StackId']
        
        # Initialize tracking variables
        seen_events = set()
        last_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        failed_reason = []
        completed_resources = []
        
        # Track the stack creation
        while True:
            # Get all new events
            events = cfn_client.describe_stack_events(StackName=stack_id)['StackEvents']
            new_events = []
            
            for event in events:
                event_id = event['EventId']
                if event_id not in seen_events and event['Timestamp'] > last_timestamp:
                    new_events.append(event)
                    seen_events.add(event_id)
            
            # Process new events in chronological order
            for event in sorted(new_events, key=lambda x: x['Timestamp']):
                resource_id = event['LogicalResourceId']
                status = event['ResourceStatus']
                reason = event.get('ResourceStatusReason', 'N/A')
                
                print(f"Resource: {resource_id}, Status: {status}, Reason: {reason}")
                
                # Track resource status and capture error messages
                if status == 'CREATE_FAILED' and resource_id not in failed_reason:
                    failed_reason.append({
                        'resource': resource_id,
                        'reason': reason
                    })
                elif status == 'DELETE_IN_PROGRESS' and reason != 'N/A' and not failed_reason:
                    # Capture validation errors from DELETE_IN_PROGRESS status when no CREATE_FAILED exists
                    failed_reason.append({
                        'resource': resource_id,
                        'reason': reason
                    })
                elif status == 'CREATE_COMPLETE' and resource_id not in completed_resources:
                    completed_resources.append(resource_id)
            
            # Check stack status
            stack = cfn_client.describe_stacks(StackName=stack_id)['Stacks'][0]
            status = stack['StackStatus']
            
            if status == 'CREATE_COMPLETE':
                # Clean up successful stack
                cfn_client.delete_stack(StackName=stack_id)
                return {
                    'success': True,
                    'error_message': None,
                    'stack_id': stack_id,
                    'failed_reason': failed_reason,
                    'completed_resources': completed_resources,
                    'stack_events': None
                }
            elif status in ['CREATE_FAILED', 'ROLLBACK_COMPLETE', 'ROLLBACK_FAILED', 'DELETE_COMPLETE']:
                # print("Returning")
                return {
                    'success': False,
                    'error_message': failed_reason[0]['reason'] if failed_reason else "Unknown error occurred",
                    'stack_id': stack_id,
                    'failed_reason': failed_reason,
                    'completed_resources': completed_resources,
                    'stack_events': list(seen_events)
                }
            
            time.sleep(2)
            
    except ClientError as e:
        return {
            'success': False,
            'error_message': str(e),
            'stack_id': None,
            'failed_reason': e,
            'completed_resources': [],
            'stack_events': None
        }
    except Exception as e:
        return {
            'success': False,
            'error_message': f"Unexpected error: {str(e)}",
            'stack_id': None,
            'failed_reason': e,
            'completed_resources': [],
            'stack_events': None
        }


# Useful - Resource Coverage
def analyze_resource_coverage(sample_template_path, LLM_generate_template_path, required_resources=None):
    """
    Analyze the percentage of resource coverage of sample template and LLM generated templates.
    Handles duplicate resources in the templates.
    """
    if required_resources is None:
        required_resources = get_required_resource_types(sample_template_path).get("resource_types")
    generated_resources = get_required_resource_types(LLM_generate_template_path).get("resource_types")

    # Count occurrences of each resource type
    required_counts = {}
    generated_counts = {}
    
    for resource in required_resources:
        required_counts[resource] = required_counts.get(resource, 0) + 1
    for resource in generated_resources:
        generated_counts[resource] = generated_counts.get(resource, 0) + 1
    
    # Calculate correct, missing, and extra resources
    all_resource_types = set(required_counts.keys()) | set(generated_counts.keys())
    correct_resources = []
    missing_resources = []
    extra_resources = []
    
    for resource_type in all_resource_types:
        req_count = required_counts.get(resource_type, 0)
        gen_count = generated_counts.get(resource_type, 0)
        
        # Add to correct resources (minimum of required and generated)
        min_count = min(req_count, gen_count)
        correct_resources.extend([resource_type] * min_count)
        
        # Add to missing resources (if required > generated)
        if req_count > gen_count:
            missing_resources.extend([resource_type] * (req_count - gen_count))
            
        # Add to extra resources (if generated > required)
        if gen_count > req_count:
            extra_resources.extend([resource_type] * (gen_count - req_count))

    metrics = {
        "total_required_resources": len(required_resources),
        "total_generated_resources": len(generated_resources),
        "correct_resources": len(correct_resources),
        "missing_resources": len(missing_resources),
        "extra_resources": len(extra_resources),
        "coverage_percentage": (len(correct_resources) / len(required_resources) * 100) if required_resources else 0,
        "accuracy_percentage": (len(correct_resources) / len(generated_resources) * 100) if generated_resources else 0,
        "resource_details": {
            "correct": correct_resources,
            "missing": missing_resources,
            "extra": extra_resources
        }
    }
    return metrics


# Process Function
def evaluate_templates_from_csv(csv_input_path, csv_output_path, llm_type):
    """
    Evaluate templates listed in a CSV file and save results to a new CSV file.
    
    Args:
        csv_input_path (str): Path to input CSV containing template paths
        csv_output_path (str): Path to save evaluation results
    """
    # Read the input CSV
    df = pd.read_csv(csv_input_path)
    
    results = []
    for _, row in df.iterrows():
        ground_truth_path = row['ground_truth_path']
        llm_template_path = row[f'{llm_type}_template_path']
        difficulty_level = row['difficulty_level']
        service = row['service']
        
        # Step 0: YAML syntax validation
        yaml_valid, yaml_error = yaml_syntax_validation(llm_template_path)
        if not yaml_valid:
            results.append({
                f'{llm_type}_template_path': llm_template_path,
                'difficulty_level': difficulty_level,
                'service': service,
                'yaml_error': yaml_error,
                'syntax_correctness': False,
                'syntax_errors': None,
                'syntax_error_details': None,
                'deploy_correctness': False,
                'failed_reason': None,
                'completed_resources': None,
                'coverage_percentage': 0,
                'accuracy_percentage': 0
            })
            continue
        
        # Step 1: Template syntax validation
        syntax_result = evaluate_template_with_linter(llm_template_path)
        
        # Format error details into a readable string
        error_messages = []
        if not syntax_result['passed']:
            for error in syntax_result['error_details']:
                error_msg = (f"Resource: {error['resource']} | "
                           f"Error: {error['message']} | "
                           f"Line: {error['line_number']} | "
                           f"Description: {error['rule_description']}")
                error_messages.append(error_msg)
        
        # Step 2: Deployment validation
        if syntax_result['passed']:
            deploy_result = evaluate_template_deployment(llm_template_path)
            failed_reason = deploy_result.get('failed_reason', None)
            completed_resources = deploy_result.get('completed_resources', None)
        else:
            deploy_result = {'success': False}
            failed_reason = None
            completed_resources = None      

        # Step 3: Resource coverage analysis
        coverage_result = analyze_resource_coverage(ground_truth_path, llm_template_path)
        
        # Compile results
        evaluation_result = {
            f'{llm_type}_template_path': llm_template_path,
            'yaml_error': None,
            'syntax_correctness': syntax_result['passed'],
            'syntax_errors': syntax_result['total_issues'],
            'syntax_error_details': '\n'.join(error_messages) if error_messages else None,
            'deploy_correctness': deploy_result['success'] if syntax_result['passed'] else "FALSE",
            'failed_reason': failed_reason,
            'completed_resources': completed_resources,
            'coverage_percentage': coverage_result['coverage_percentage'],
            'accuracy_percentage': coverage_result['accuracy_percentage'],
            'missing_resources': coverage_result['resource_details']['missing'],

        }
        results.append(evaluation_result)
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df


def main():
    # This function is used to validate all LLM generated template with template file path in a CSV file.
    # This main() function is not used for IaCGen, this is just to test the functions above.
    # Just run the main.py function to execute IaCGen.
    input_csv = "Data/iac.csv"
    output_csv = "Result/claude_results.csv"
    llm_type = "claude"   # gemini, gpt, claude
    results = evaluate_templates_from_csv(input_csv, output_csv, llm_type)
    print(f"Evaluation completed. Results saved to {output_csv}")


if __name__ == "__main__":
    print("Start Evaluation")
    main()
    print("End Evaluation")


