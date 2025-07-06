import os
import json
import subprocess
from configparser import ConfigParser

# Define AWS credentials path
AWS_CREDENTIALS_PATH = os.path.join(os.path.expanduser("~"), ".aws", "credentials")

# Define roles and session names for both profiles
ROLES = {
    "default": {
        "role_arn": "arn:aws:iam::944287611987:role/application-engineer",
        "session_name": "application-engineer-session",
    },
    "test": {
        "role_arn": "arn:aws:iam::944287611987:role/app-bedrock-access-900858-us-east-1/test",
        "session_name": "test-session",
    },
}

def assume_role(role_arn, session_name):
    """Runs AWS CLI assume-role command and returns credentials."""
    try:
        cmd = (
            f"aws sts assume-role --role-arn {role_arn} "
            f"--role-session-name {session_name} --output json"
        )
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
        credentials = json.loads(result.stdout)['Credentials']
        return credentials
    except subprocess.CalledProcessError as e:
        print(f"Error assuming role: {e.stderr}")
        return None

def update_aws_credentials(profile_name, credentials):
    """Updates the AWS credentials file with new temporary credentials."""
    config = ConfigParser()

    # Read existing credentials
    if os.path.exists(AWS_CREDENTIALS_PATH):
        config.read(AWS_CREDENTIALS_PATH)

    # Add or update profile
    if profile_name not in config:
        config.add_section(profile_name)

    config[profile_name]['aws_access_key_id'] = credentials['AccessKeyId']
    config[profile_name]['aws_secret_access_key'] = credentials['SecretAccessKey']
    config[profile_name]['aws_session_token'] = credentials['SessionToken']

    # Write updated credentials back
    with open(AWS_CREDENTIALS_PATH, 'w') as configfile:
        config.write(configfile)

    print(f"AWS credentials updated for profile: {profile_name}")

if __name__ == "__main__":
    for profile, role_info in ROLES.items():
        credentials = assume_role(role_info["role_arn"], role_info["session_name"])
        if credentials:
            update_aws_credentials(profile, credentials)



