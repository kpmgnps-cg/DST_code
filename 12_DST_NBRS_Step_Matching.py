import pandas as pd
import pickle
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import datetime
import logging
import gzip,json
import traceback
import warnings,os
from fuzzywuzzy import fuzz

tfidf_nbrsPath = config['tfidf_nbrsPath']
tfidf_vectorizerPath = config['tfidf_vectorizerPath']

def loadmodels():
    global tfidf_vectorizer,tfidf_nbrs
    with gzip.open(tfidf_vectorizerPath, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with gzip.open(tfidf_nbrsPath, 'rb') as file:
        tfidf_nbrs = pickle.load(file)
    return "Loaded trained models for prediction"

# def splitDataFrameIntoSmaller(DST_all_org, numberChunks):
#     listOfDf = list()
#     chunkSize = len(DST_all_org) // numberChunks
#     rem=len(DST_all_org) % numberChunks
#     if rem > 0:
#         for i in range(numberChunks+1):
#             listOfDf.append(DST_all_org[i*chunkSize:(i+1)*chunkSize])
#     else:
#         for i in range(numberChunks):
#             listOfDf.append(DST_all_org[i*chunkSize:(i+1)*chunkSize])
#     return listOfDf
  
def splitDataFrameIntoSmaller(DST_all_org, numberChunks):
    listOfDf = list()
    chunkSize = len(DST_all_org) // numberChunks
    rem=len(DST_all_org) % numberChunks
    if rem > 0:
        for i in range(numberChunks+1):
          if i<numberChunks:
            listOfDf.append(DST_all_org[i*chunkSize:(i+1)*chunkSize])
          else:
            listOfDf.append(DST_all_org[i*chunkSize:])
    else:
        for i in range(numberChunks):
            listOfDf.append(DST_all_org[i*chunkSize:(i+1)*chunkSize])
    return listOfDf  
  

def getNearesttopN(query,vectorizer,nbrs,neighbors=5):
    try:
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_, n_neighbors=neighbors)
    except Exception:
        queryTFIDF_ = queryTFIDF_.reshape(1, vectorizer.vocabulary_)
        distances, indices = nbrs.kneighbors(queryTFIDF_, n_neighbors=neighbors)
    return distances, indices

def applyChunks3(typer,vectorizer,nbrs,company_names, partyid,c):
    results = []
    global config
    try:
        if typer !='address_':
            elements = c['allReg']
        else:
            elements = c['allReg']
        if typer=="ngram_":
            neighbors=config["ngram_neighbors"]
        else:
            neighbors=config["neighbors"]
        distances, indices = getNearesttopN(elements,vectorizer,nbrs,neighbors)
        unique_org = list(elements)
        matches = []
        if len(indices)>0:
            cols=["source_record"]
            for i in range(0,neighbors):
                cols=cols+['match'+str(i),'score'+str(i),'partyid'+str(i)]
            for i,j in enumerate(indices):
                temp=[]
                temp.append(unique_org[i])
                for k in range(0,neighbors):
                    temp.append(company_names.values[j][k])
                    temp.append(round(distances[i][k],2))
                    temp.append(int(partyid.values[j][k]))
                matches.append(temp)
        results = pd.DataFrame(matches, columns=cols)
    except Exception:
        logger.error(traceback.format_exc())
        results =pd.DataFrame(columns=cols)
    return results

def partyid_to_plansponsor(partyid,matchData,metadata):
    party=list(set(list(matchData.loc[metadata['target']['primaryKey']==partyid,metadata['target']['matchColumns']])))
    if len(party)>0:
        return party[0]
    else:
        return ""

def scoreResults3(results, typer,neighbors=5):
    logger.info("Generating scores using fuzzy matching for the identifed matches .... ")
    global config
    if typer=="ngram_":
        neighbors=config["ngram_neighbors"]
    else:
        neighbors=config["neighbors"]
    for i in range(0,neighbors):
        results[typer+ "totalScore"+str(i)] = 1 - results["score"+str(i)].replace(0.00,0.01)
        #results[typer+ "totalScore"+str(i)] = fuzz.token_sort_ratio(results['match'+str(i)],results["source_record"])
        results.drop(columns=["score"+str(i)],axis=1,inplace=True)
        # Entity
        results[typer + 'matched_entity'+str(i)] = results['match'+str(i)]
        results.drop(columns=["match"+str(i)],axis=1,inplace=True)
        # ID
        results[typer+'matched_id'+str(i)] = results['partyid'+str(i)]
        results.drop(columns=["partyid"+str(i)],axis=1,inplace=True)
    return results

def trainmatching(dataset,matchData,matchNames,config,customEntities,metadata,ncores):
    try:
        results_main=pd.DataFrame()
        logger.info("Loading ML Search Models")
        logging.info("Fitting count vectorizers and preparing nearest neighbour model .... ")
        partyid=metadata['target']['primaryKey']
        #ML 1 - Word count
        if ncores>dataset.shape[0]:
            ncores=dataset.shape[0]
        matchNames = matchData[metadata['target']['matchColumns']].fillna('xx')
        partyids=matchData[partyid].fillna('-1')
        logger.info("matchNames ={},partyids={}".format(str(len(matchNames)),str(len(partyids))))
        pool = Pool(ncores)
        chunks = splitDataFrameIntoSmaller(dataset, ncores)

        #ML 3 - Word weight model
        logging.info("Applying fitted model=word weight on source data .... ")
        typer = 'tfidf_'
        funcs = partial(applyChunks3, typer,tfidf_vectorizer,tfidf_nbrs,matchNames, partyids)
        results3 = pool.map(funcs, chunks)
        results3 = pd.concat(results3)
        results_main=scoreResults3(results3,typer).reset_index(drop=True)

        #results_main = pd.concat([results2,results3], axis=1)
        results_main = results_main.loc[:,~results_main.columns.duplicated()]
        pool.close()
    except Exception as e:
        if pool:
            pool.close()
        logger.error(traceback.format_exc())
    return results_main

def matching(dataset,matchData,matchNames,config,customEntities,metadata,ncores):
    # Load Nearest Neighbors and count models
    results_main=pd.DataFrame()
    now = datetime.datetime.now()
    data=dataset.copy()
    try:
        results_main=trainmatching(dataset,matchData,matchNames,config,customEntities,metadata,ncores)
        results_main["allReg"]=data["allReg"]
        logger.info("time took ={} ".format(str(datetime.datetime.now()-now)))
        #logger.info(results_main.to_json(orient="records"))
    except Exception:
        logger.error(traceback.format_exc())
    return results_main