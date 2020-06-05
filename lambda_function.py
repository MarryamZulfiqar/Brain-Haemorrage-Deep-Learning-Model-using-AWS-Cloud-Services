import json
import urllib
import urllib.parse
from urllib.parse import urlparse
import boto3
from PIL import Image
import PIL.Image
import os
import sys
import uuid

print('Loading function')

s3 = boto3.client('s3')


def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        url = '{}/{}/{}'.format(s3.meta.endpoint_url, bucket, key)
        print("URL: " + url)
        print("CONTENT TYPE: " + response['ContentType'])
    
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
        im = s3.download_file(bucket, key, download_path)

    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
