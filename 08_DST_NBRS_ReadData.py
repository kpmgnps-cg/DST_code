import pandas as pd
import json
import re

def normalizer(x):
    chars_to_remove = [")","(",".","|","[","]","{","}","'",",","&","","_","-","/","#","*","`"]
    x=str(x).upper().replace(",INC",", INC").replace(",LLC",", LLC").replace(",PC",", PC").replace("INC.","INC. ").replace("LLC.","LLC. ").replace("PC.","PC. ")    
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, ' ', x)
    string= re.sub(' +',' ',string).upper().strip()
    return string

def convert(x):
    try:
        x = int(x)
    except Exception as e:
        try:
            x = float(x)
        except Exception as e:
            x = x
    return x

def readMatchLookup(filepath,header=list(),matchColumns=list(),primaryKey=list()):
    if len(header)==0:
        matchData= pd.read_csv(filepath,engine='python')
        matchData.columns=map(str.lower,matchData.columns)
    else:
        matchData= pd.read_csv(filepath,header=None,engine="python")
        matchData.columns=header  
        matchData.columns=map(str.lower,matchData.columns)  
    if matchColumns:
        #matchData.sort_values(primaryKey, inplace=True)
        if primaryKey:
            matchData[matchColumns] = matchData[matchColumns].apply(lambda x : str(x).upper())
            matchData[matchColumns] = matchData[matchColumns].fillna("")
            matchData[primaryKey] = matchData[primaryKey].fillna(-99999)
            matchData[primaryKey] =  matchData[primaryKey].apply(lambda x: convert(x))
            matchNames = matchData[matchColumns]         
    else:
        print("Must specify the column name/names to be used in matching")
    return matchData,matchNames
  
def readSource(config,customEntities,metadata,flag="train"):
  logger.info('Loading training data')
  Dataset=pd.DataFrame()
  try:
      if config["sourcedbtype"]=="file":
          Dataset = pd.read_csv(config["source"])
          Dataset.columns=map(str.lower,Dataset.columns)          
      else:
          logger.info("Not a valid source type")
      if Dataset.shape[0]>0:
          matchColumns=metadata['source']['matchColumns']
          Dataset['allReg'] = ' '
          for i in matchColumns:
              Dataset[i.lower()].fillna("",inplace=True)
              Dataset[i.lower()]=Dataset[i.lower()].apply(lambda x:str(x).strip())
              Dataset['allReg']=Dataset['allReg']+Dataset[i.lower()] + ' '   
          Dataset['allReg'] = Dataset['allReg'].fillna('xx')
          Dataset['allReg'] = Dataset['allReg'].apply(lambda x: x.upper().strip())
          Dataset['allReg'] = Dataset['allReg'].apply(lambda x: re.sub(' +',' ',x))
          #Dataset = Dataset.drop_duplicates(subset=['cum_dsc_num','acct_num'])
          Dataset = Dataset[Dataset['allReg'].notnull()]  
          Dataset = Dataset[Dataset['allReg']!='xx'] 
          logger.info('training data shape = {}'.format(str(Dataset.shape)))
  except Exception:
      logger.error(traceback.format_exc())    
  return Dataset

def readTarget(config,customEntities,metadata):
    logger.info("Reading the match source look up data for AI matching .... ")
    matchData, matchNames=pd.DataFrame(),list()
    try:
        if config["matchdbtype"]=="file":
            matchData, matchNames = readMatchLookup(config['matchdata'],header=[],
                                                   matchColumns=metadata['target']['matchColumns'],primaryKey=metadata['target']['primaryKey'])
        logger.info("match data shape ={}".format(str(matchData.shape)))    
        matchNames = matchNames.fillna('xx')
    except Exception:
        logger.error(traceback.format_exc())    
    return matchData, matchNames