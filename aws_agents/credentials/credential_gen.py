import boto3
import configparser
import os

# Define the role to assume
ROLE_ARN = "arn:aws:iam::942286715197:role/100866-application-engineer"
SESSION_NAME = "auto-session"

# Initialize STS client
sts_client = boto3.client("sts")

# Assume the role
response = sts_client.assume_role(
    RoleArn=ROLE_ARN,
    RoleSessionName=SESSION_NAME
)

print("response",response)