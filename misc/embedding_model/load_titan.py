# # "Action": [
# #       "bedrock:GetInferenceProfile",
# #       "bedrock:ListInferenceProfiles",
# #       "bedrock:DeleteInferenceProfile"
# #       "bedrock:TagResource",
# #       "bedrock:UntagResource",
# #       "bedrock:ListTagsForResource"
# #   ]


# !pip install --upgrade --force-reinstall boto3 botocore awscli

# #Check the latest version of boto3
# import boto3
# print(boto3.__version__)

# #Create Bedrock and Bedrock runtime clients
# bedrock = session.client("bedrock", region_name=studio_region)
# br = session.client("bedrock-runtime", region_name=studio_region)


# inf_profile_arn = " "  # add your profile arn


# system_prompt = "You are an expert on AWS services and always provide correct and concise answers."
# input_message = "Should I be storing documents in Amazon S3 or EFS for cost effective applications?"
# start = time()
# response = br.converse(
#     modelId=inf_profile_arn,
#     system=[{"text": system_prompt}],
#     messages=[{
#         "role": "user",
#         "content": [{"text": input_message}]
#     }]
# )




# import json
# body = json.dumps({
#     "anthropic_version": "bedrock-2023-05-31",
#     "max_tokens": 1024,
#     "temperature": 0.1,
#     "top_p": 0.9,
#     "system": system_prompt,
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": f"{input_message}",
#                 }
#             ]
#         }
#     ]
# })
# accept = 'application/json'
# contentType = 'application/json'
# response = br.invoke_model(body=body, modelId=inf_profile_arn, accept=accept, contentType=contentType)
# response_body = json.loads(response.get('body').read())