#Cyberark Authentication Code

import requests
import json
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import yaml,os

def get_credentials(query_address, appID):
	ark_url = 'https://CPZ-CCP.CAPGROUP.COM/AIMWebService/api/Accounts'
	query_params = {
		'AppID':appID,
		'Query':query_address,
		'QueryFormat':'Regexp'
		}
	r = requests.get(url=ark_url,params=query_params, verify='/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/Config/symantec.pem')
	data = r.json()
	return data['Content']