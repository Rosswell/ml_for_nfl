import boto3
import botocore
from io import StringIO
import pandas as pd


class S3Connection:
    """Establishes a connection to a Postgres DB, in order to retrieve data for training.

    Args:
        aws_id (str): AWS account id
        aws_secret (str): AWS account secret

    """
    def __init__(self, aws_id, aws_secret):
        self.aws_id = aws_id
        self.aws_secret = aws_secret

    def s3_data_to_df(self, bucket_name, bucket_key):

        """Imports s3 data and converts into a pandas dataframe.

        Args:
            bucket_name (str): S3 bucket name
            bucket_key (str): path to data within the S3 bucket

        Returns:
            pandas dataframe: if the connection was successful, the s3 data object will have been converted to a pandas
            dataframe and returned.

        Raises:
            ClientError: if the connection to the s3 bucket was unsuccessful, and was not due to a 404 response.

        """
        try:
            s3 = boto3.client('s3', aws_access_key_id=self.aws_id,
                              aws_secret_access_key=self.aws_secret)

            bucket_object = s3.get_object(Bucket=bucket_name, Key=bucket_key)
            body = bucket_object['Body']
            csv_string = body.read().decode('utf-8')
            return pd.read_csv(StringIO(csv_string))

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
