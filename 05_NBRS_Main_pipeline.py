import json
import os,sys,ast
from typing import List
import traceback
import warnings
import numpy as np
import multiprocessing as mp
import datetime
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


configPath = args["config_folder"]+'config_app.json'
customEntitiesPath=args["config_folder"]+'customEntities.json'
customEntitiesPath_RPO=args["config_folder"]+'customEntities_RPO.json'
metadataPath=args["config_folder"]+'metadata.json'
 
with open(configPath) as json_file:
    config = json.load(json_file)
    
if (file_type=="RPO")|(file_type=="Schwab"):
  print(customEntitiesPath_RPO)
  with open(customEntitiesPath_RPO) as json_file:
      customEntities = json.load(json_file)  
else:
  print(customEntitiesPath)
  with open(customEntitiesPath) as json_file:
      customEntities = json.load(json_file)
    
with open(metadataPath) as json_file:
    metadata = json.load(json_file)
	
config={"sourcedbtype": "file",
"source":"/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/input_data_latest.csv",
"chunks":5000,
"matchdbtype": "file",
"matchSource":"CDM",
"matchColumn_allplan":"org_name",
"matchdata": "/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/cdm_all_plan_sponsors_latest.csv",
"allplans_dump":"/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/all_plans_latest.csv",
"regexFilter":"True",

"ngrams_nbrsPath":"nbrs_nn.pickle",
"ngrams_vectorizerPath":"vectorizer_nn.pickle",

"tfidf_nbrsPath":"nbrs_tfidf.pickle",
"tfidf_vectorizerPath":"vectorizer_tfidf.pickle",

"entitiesPath":"/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/customEntities.json",
"metadataPath":"/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/metadata.json",
"mainpipelineflag":"predict",
"ncores":15,
"neighbors":15,
"ngram_neighbors":1
}


