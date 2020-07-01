import os
import json
import numpy as np
import urllib.request
import boto3
import psycopg2
from datetime import datetime
import sagemaker
from sagemaker.predictor import RealTimePredictor


def lambda_handler(event=None, context=None):
  try:
    # Downloading New Case
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    new_case_url = '{}/{}/{}'.format(s3.meta.endpoint_url, bucket, key)
    # new_case_url = 'https://res.cloudinary.com/itu/image/upload/v1592184341/100055CT000270_wuhm1w.png'
    file_name = new_case_url.split('/')[-1]
    payload = urllib.request.urlopen(new_case_url).read()

    # Predicting 
    prediction = predict(payload)

    if prediction ==  'Intracranial':
      prediction_record = (file_name, 'True', 'False', 'False', 'Intracranial Hemorrhage', datetime.now())
    elif prediction ==  'Mass Effect':
      prediction_record = (file_name, 'False', 'True', 'False', 'Mass Effect', datetime.now())
    elif prediction ==  'Midline Shift':
      prediction_record = (file_name, 'False', 'False', 'True', 'Midline Shift', datetime.now())
 
    # Inserting into database
    insert_in_database(prediction_record)

    print('Prediction successfully saved in Database!')
  
    return 'Success'
  except Exception as ex:
    return ex


def predict(payload):
   # Configurations
   region = os.environ['REGION'] 
   access_id = os.environ['ACCESS_ID']
   secret_key = os.environ['SECRET_KEY']
   endpoint = 'brain-model-ep--2020-06-16-14-46-14'
   boto_session = boto3.Session(region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
   session = sagemaker.Session(boto_session=boto_session)
   predictor = RealTimePredictor(endpoint=endpoint, sagemaker_session=session, serializer=None)
   class_mappings = ['Intracranial', 'Mass Effect', 'Midline Shift'] 

   # Predicting
   prediction = json.loads(predictor.predict(payload))
   return  class_mappings[np.argmax(prediction)]


def insert_in_database(prediction_record):
    conn = None
    try:
        # Configurations
        db_user = os.environ['DB_USER']
        db_password = os.environ['DB_PASSWORD'] 
        db_host = os.environ['DB_HOST'] 
        db_port = os.environ['DB_PORT'] 

        # Establishing Connection
        conn = psycopg2.connect(user = db_user, password = db_password, host = db_host, port = db_port)
        conn.autocommit = True

        # Query Definitions
        insert_query = 'INSERT INTO prediction (file_name, intracranial_hemorrhage, mass_effect, midline_shift, prediction, prediction_date) VALUES(%s,%s,%s,%s,%s,%s)'

        # Record Insertion
        cursor = conn.cursor()
        cursor.execute(insert_query, prediction_record)
        
        return 'Success'
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn:
            conn.close()