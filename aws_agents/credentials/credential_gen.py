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

#link to the tutorial - https://bobbyhadz.com/blog/install-and-use-jq-on-windows

import subprocess
import json
import configparser
import os

# Define role ARN and session name
ROLE_ARN = "arn:aws:iam::942286715197:role/108866-application-engineer"
SESSION_NAME = "app-engineer-session"

# Get AWS credentials file path for Windows
AWS_CREDENTIALS_PATH = os.path.join(os.getenv("USERPROFILE"), ".aws", "credentials")

def assume_role():
    """Runs AWS CLI assume-role command and returns credentials."""
    try:
        cmd = [
            "aws", "sts", "assume-role",
            "--role-arn", ROLE_ARN,
            "--role-session-name", SESSION_NAME,
            "--output", "json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        credentials = json.loads(result.stdout)['Credentials']
        return credentials
    except subprocess.CalledProcessError as e:
        print("Error assuming role:", e.stderr)
        return None

def update_aws_credentials(credentials, profile_name="assumed-role"):
    """Updates the AWS credentials file with new temporary credentials."""
    config = configparser.ConfigParser()

    # Read existing credentials
    if os.path.exists(AWS_CREDENTIALS_PATH):
        config.read(AWS_CREDENTIALS_PATH)

    if profile_name not in config:
        config.add_section(profile_name)

    config[profile_name]['aws_access_key_id'] = credentials['AccessKeyId']
    config[profile_name]['aws_secret_access_key'] = credentials['SecretAccessKey']
    config[profile_name]['aws_session_token'] = credentials['SessionToken']
    
    # Write updated credentials back
    with open(AWS_CREDENTIALS_PATH, 'w') as configfile:
        config.write(configfile)
    
    print(f"AWS credentials updated under profile: [{profile_name}]")

if __name__ == "__main__":
    credentials = assume_role()
    if credentials:
        update_aws_credentials(credentials)

