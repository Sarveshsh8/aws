# import boto3
# import configparser
# import os

# # Define the role to assume
# ROLE_ARN = "arn:aws:iam::942286715197:role/100866-application-engineer"
# SESSION_NAME = "auto-session"

# # Initialize STS client
# sts_client = boto3.client("sts")

# # Assume the role
# response = sts_client.assume_role(
#     RoleArn=ROLE_ARN,
#     RoleSessionName=SESSION_NAME
# )

# print("response",response)



import subprocess
import json
import configparser
import os

# Define role ARN and session name
ROLE_ARN = "add your arn number"
SESSION_NAME = "session name"

def assume_role():
    """Runs AWS CLI assume-role command and returns credentials."""
    try:
        cmd = [
            "aws", "sts", "assume-role",
            "--role-arn", ROLE_ARN,
            "--role-session-name", SESSION_NAME,
            "--output", "json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        credentials = json.loads(result.stdout)['Credentials']
        return credentials
    except subprocess.CalledProcessError as e:
        print("Error assuming role:", e.stderr)
        return None

def update_aws_credentials(credentials, profile_name="assumed-role"):
    """Updates the ~/.aws/credentials file with new temporary credentials."""
    aws_credentials_path = os.path.expanduser("~/.aws/credentials")
    
    config = configparser.ConfigParser()
    config.read(aws_credentials_path)

    if profile_name not in config:
        config.add_section(profile_name)

    config[profile_name]['aws_access_key_id'] = credentials['AccessKeyId']
    config[profile_name]['aws_secret_access_key'] = credentials['SecretAccessKey']
    config[profile_name]['aws_session_token'] = credentials['SessionToken']
    
    with open(aws_credentials_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"AWS credentials updated under profile: [{profile_name}]")

if __name__ == "__main__":
    credentials = assume_role()
    if credentials:
        update_aws_credentials(credentials)
