# -*- coding: utf-8 -*-
"""
Created on Jul 2021 


"""
import re,os
import numpy as np
import pandas as pd
import multiprocessing as mp
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")
from fuzzywuzzy import fuzz

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses,util
from torch.utils.data import DataLoader
from sentence_transformers import  models
from torch import nn
modelpath="./custommodel-cosine-fulldata"

import torch
torch.cuda.empty_cache()

replacements_strip = {r'\bLLC\b':'',r'\bLLP\b':'',r'\bCORP\b':'',
                r'\bINC\b':'',
                r'\bPLLC\b':'',r'\bPA\b':'',r'\bPLC\b':'',r'\bPC\b':''}

#modelpath=dbutils.widgets.get("modelpath")
#model_name=dbutils.widgets.get("model_name")
cuda_device='cuda:2'
model = SentenceTransformer(modelpath,device=cuda_device)


matchdata = pd.read_csv(config["matchdata"])
matchdata=matchdata.sort_values(["street_line_1","street_line_2","city","state2","postal_code",
                               "country2"]).drop_duplicates(["partyid","org_name","street_line_1","street_line_2","city","state2","postal_code",
                                                            "country2"])
matchdata['count']=matchdata.groupby(["partyid","org_name"])["partyid"].transform('count')
matchdata=matchdata.loc[(matchdata['count']==1)|((matchdata['count']>1)&((pd.notnull(matchdata['city']))|(pd.notnull(matchdata['state2']))|(pd.notnull(matchdata['postal_code']))|(pd.notnull(matchdata['country2']))))].reset_index(drop=True)
plan_sponsor=[str(x).strip().upper() for x in list(matchdata["org_name"])]

plan_sponsor=[str(x).strip().upper() for x in list(matchdata["org_name"])]
party_id_list = list(matchdata['partyid'])

cdm_party_id_to_org={}
cdm_org_to_party_id={}
for party_id,plan_spnsr in zip(party_id_list,plan_sponsor):
  try :
    cdm_party_id_to_org[int(party_id)] = str(plan_spnsr).upper().strip()
    cdm_org_to_party_id[str(plan_spnsr).upper().strip()]=int(party_id)
  except Exception as e:
    print(e)
	

all_plans_org = pd.read_csv(config["allplans_dump"])
print(all_plans_org.shape)
all_plans_org=all_plans_org[all_plans_org.PARTY_STATUS=='Active']
all_plans_org["org_name_proc"]=all_plans_org["org_name"].apply(lambda x:normalizer(str(x).upper()))
all_plans_org=all_plans_org.loc[all_plans_org.plan_name.notnull(),:].reset_index(drop=True)
all_plans_org["org_name"]=all_plans_org["org_name"].apply(lambda x:re.sub(' +',' ',str(x)).upper().strip() if len(str(x))>0 else x)
all_plans_org["plan_name"]=all_plans_org["plan_name"].apply(lambda x:re.sub(' +',' ',str(x)).upper().strip() if len(str(x))>0 else x)
all_plans_org=all_plans_org.sort_values(["org_name","plan_value_date"],ascending=[True,False]).reset_index(drop=True)
print(all_plans_org.shape)

plan_name_list=[str(x).strip().upper() for x in list(all_plans_org["plan_name"])]
allplan_partyid_list = list(all_plans_org['party_id'])
org_name_list = list(all_plans_org['org_name'])
plan_id_list = list(all_plans_org['plan_id'])
org_name_proc_list = list(all_plans_org['org_name_proc'])

allplan_planname_partyid={}
allplan_party_id_to_org={}
allplan_process_org_partyid={}
allplan_planname_planid={}
allplan_planid_planname={}
allplan_org_partyid={}
for party_id,plan_name,org_name,org_name_proc,plan_id in zip(allplan_partyid_list,plan_name_list,org_name_list,org_name_proc_list,plan_id_list):
  try :
    allplan_planname_partyid[plan_name] = int(party_id)   
    allplan_process_org_partyid[str(org_name_proc)] = int(party_id)
    allplan_party_id_to_org[int(party_id)] = str(org_name).upper().strip()
    allplan_planname_planid[plan_name]=plan_id
    allplan_planid_planname[plan_id]=plan_name
    allplan_org_partyid[org_name] = int(party_id)
  except Exception as e:
    print(e)

#Cell added by Ganapathy
all_plans_org["plantype"]=all_plans_org.apply(lambda x: re.sub("\s+"," ",x["plan_name"].replace(x["org_name"],"").replace(".","").replace(",","").replace("INC","").replace("LLC","").replace("401K","401(K)").replace("401 K","401(K)").replace("401 (K)","401(K)").replace("401-K","401(K)").replace(".401","401").replace("C401","401").replace("S401","401").replace("S 401","401").replace("I 401","401").replace("P401","401").replace("C 401","401").replace("-401","401").replace("- 401","401").replace("P.","").replace("L401","401").replace("MD "," ").replace("PC PROFIT","PROFIT").replace("403 B","403 (B)").replace("403 (B)","403(B)").replace("-"," ").replace("PS PLAN","PROFIT SHARING PLAN").replace("P/S PLAN","PROFIT SHARING PLAN").replace("P/S","PROFIT SHARING PLAN").replace("RET PLAN","RETIREMENT PLAN").replace("PSP","PROFIT SHARING PLAN").replace("P S PLAN","PROFIT SHARING PLAN").replace("CCASH BALANCE PLAN","CASH BALANCE PLAN").replace("CRETIREMENT PLAN","RETIREMENT PLAN").replace("IN "," ").replace("PA PROFIT","PROFIT").replace("LTD401","401").replace("PA ","").replace("LLP ","").strip()) if len(x["plan_name"].replace(x["org_name"],""))<len(x["plan_name"]) else "",axis=1)

plantype_list=list(all_plans_org.plantype.value_counts()[all_plans_org.plantype.value_counts()>10].index)
plantype_list=[i for i in plantype_list if i !=""]
plantype_list=[e for e in plantype_list if e not in ('PLAN', 'THE 401(K) PLAN', '.', 'C', 'LC', 'PLAN & TRUST', 'NC', 'PROFIT', 'N', '401(K PLAN', 'C.' , 'S', 'AN', 'THE', 'PC', '401(K', "'401(K) PLAN",'LAN', 'P','401(K) PLA')]


def fetch_plan_type(record):
  record=re.sub("\s+"," ",record.replace(".","").replace(",","").replace("INC","").replace("LLC","").replace("401K","401(K)").replace("401 K","401(K)").replace("401 (K)","401(K)").replace("401-K","401(K)").replace(".401","401").replace("C401","401").replace("S401","401").replace("S 401","401").replace("I 401","401").replace("P401","401").replace("C 401","401").replace("-401","401").replace("- 401","401").replace("P.","").replace("L401","401").replace("MD "," ").replace("PC PROFIT","PROFIT").replace("403 B","403 (B)").replace("403 (B)","403(B)").replace("-"," ").replace("PS PLAN","PROFIT SHARING PLAN").replace("P/S PLAN","PROFIT SHARING PLAN").replace("P/S","PROFIT SHARING PLAN").replace("RET PLAN","RETIREMENT PLAN").replace("PSP","PROFIT SHARING PLAN").replace("P S PLAN","PROFIT SHARING PLAN").replace("CCASH BALANCE PLAN","CASH BALANCE PLAN").replace("CRETIREMENT PLAN","RETIREMENT PLAN").replace("IN "," ").replace("PA PROFIT","PROFIT").replace("LTD401","401").replace("PA ","").replace("LLP ","").strip())
  list_matches=[i for i in plantype_list if i in record]
  if len(list_matches)>0:
    max_len=0
    match=""
    for i in list_matches:
      if len(i)>max_len:
        max_len=len(i)
        match=i
    match=re.sub("^[a-zA-Z] 401|^[a-zA-Z]401", "401",match)
    match=re.sub("^[a-zA-Z] PROFIT SHARING PLAN|^[a-zA-Z]PROFIT SHARING PLAN","PROFIT SHARING PLAN",match)
    match=re.sub("PROFIT SHARING PLAN[ &a-zA-Z]{1,}","PROFIT SHARING PLAN TRUST",match)
    return match
  else:
    return ""    


all_plans_org["plantype"]=all_plans_org.apply(lambda x: fetch_plan_type(str(x['plan_name']).strip().upper()),axis=1)


# df_test=pd.read_csv("/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/pipeline_december.csv")
# match_test=pd.read_csv("/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/aimatch_pipeline_december.csv")

df_test=pd.read_csv(args["input_folder"]+filename+".csv")
match_test=pd.read_csv(args["output_folder"]+"aimatch_"+filename+".csv")

if "party_id" in list(df_test.columns):
  df_test.rename(columns={"party_id":"PARTYID"},inplace=True) 
if "partyid" in list(df_test.columns):
  df_test.rename(columns={"partyid":"PARTYID"},inplace=True) 
if "PARTY_ID" in list(df_test.columns):
  df_test.rename(columns={"PARTY_ID":"PARTYID"},inplace=True)
if ("PARTY_ID" not in list(df_test.columns))&("PARTYID" not in list(df_test.columns)):
  df_test["PARTYID"]=None  

print(df_test.shape)
print(match_test.shape)


def postalcode_to_plansponsor(allReg,city,state,postal):
  try:
    if postal!="":
      postal=str(int(float(str(postal))))
  except Exception as e:
    postal=str(postal)
    
  try:
    if postal=="":
      return ["","",0]
    elif (pd.isna(postal))|(pd.isnull(postal))|(postal is None):
      return ["","",0]
    else:  
      add_full_t=str(postal+" "+city+" "+state).upper()
#       org_postalcode=matchData.loc[(matchData.add_full==add_full_t),["partyid","org_name"]] 
      org_postalcode=matchData.loc[(matchData.add_full==add_full_t)|(matchData.add_full_t==add_full_t),["partyid","org_name"]]      
      
      if len(org_postalcode)>0:
        org_postalcode["fuzz_score"]=org_postalcode.apply(lambda row:fuzz.token_set_ratio(str(row['org_name']).upper(),allReg)/100.0,axis=1)
        org_postalcode=org_postalcode.sort_values(["fuzz_score"],ascending=[False]).reset_index(drop=True)
        if (file_type=="Fidelity"):
          org_postalcode=org_postalcode.loc[org_postalcode.fuzz_score>=0.95,:].reset_index(drop=True)
        elif (file_type=="Schwab"):
          org_postalcode=org_postalcode.loc[org_postalcode.fuzz_score>=0.95,:].reset_index(drop=True)
        else:
          org_postalcode=org_postalcode.loc[org_postalcode.fuzz_score>=0.75,:].reset_index(drop=True) 
          
        if len(org_postalcode)>0:
          org_name=org_postalcode.loc[0,"org_name"]
          partyid=org_postalcode.loc[0,"partyid"]
          fuzz_score=org_postalcode.loc[0,"fuzz_score"]
          return [str(org_name).upper(),partyid,fuzz_score]
        else:
          return ["","",0]      
      else:
        return ["","",0]
  except Exception as e:
    return ["","",0]


def partyid_to_plansponsor(row,column,flag=False):
  if flag:
    if (row["allplan_allReg_id"]!=-9999)&(row[column] in allplan_party_id_to_org.keys()):
      return str(allplan_party_id_to_org[row[column]]).upper().strip()
    if (row["partyid_exact_allReg"]!=-9999)&(row[column] in cdm_party_id_to_org.keys()):
      return str(cdm_party_id_to_org[row[column]]).upper().strip()
    
  if row[column] in allplan_party_id_to_org.keys():
    return str(allplan_party_id_to_org[row[column]]).upper().strip()
  elif row[column] in cdm_party_id_to_org.keys():
    return str(cdm_party_id_to_org[row[column]]).upper().strip()  
  elif row[column] in original_party_id_to_org.keys():
    return str(original_party_id_to_org[row[column]]).upper().strip()    
  else:
    return ""
  
def ct(r):
  try:
    if str(r).strip()=="":
      return np.nan
    elif pd.isna(r):
      return r
    else:
      return int(float(str(r).strip()))
  except Exception as e:
    print("value is r={}".format(r))
    print(e)


match_test["allReg"]=match_test["allReg"].apply(lambda x:"" if pd.isnull(x) else x)
df_test["Groundtruth"]=df_test.apply(lambda x:partyid_to_plansponsor(x,"PARTYID"),axis=1)
df_test["allReg"]=match_test["allReg"]
if file_type=="DST":
  df_test["allRegc"]=match_test["allRegc"]
df_test["Reg_concatanate"]=match_test["Reg_concatanate"]

matchData["add_full"]=matchData['postal_code']+" "+matchData['city']+" "+matchData['state2']
matchData["add_full"]=matchData["add_full"].str.upper()

matchData["add_full_t"]=matchData['postal_code'].str[0:5]+" "+matchData['city']+" "+" "+matchData['state2']
matchData["add_full_t"]=matchData["add_full_t"].str.upper()
  
addr_match_indexlist=list()
if file_type=="DST":
  logger.info("Adddress match step is in progress")
  df_test["city"]=match_test["city"]
  df_test["StateName"]=match_test["StateName"]
  df_test["postal_code"]=match_test["postal_code"]
  match_test.postal_code=match_test.postal_code.astype(str)
  address_org=match_test[["allReg","city","StateName","postal_code"]].apply(lambda row: postalcode_to_plansponsor(row['allReg'],str(row['city']),str(row['StateName']),str(row['postal_code'])),axis=1)
  address_org=pd.DataFrame(list(address_org),columns=["tfidf_matched_entity_addr","tfidf_matched_id_addr","tfidf_totalScore_addr"])
  match_test=pd.concat([match_test,address_org],axis=1)
  addr_match_indexlist=match_test.loc[match_test["tfidf_totalScore_addr"]>0,:].reindex(columns=["allReg","Groundtruth","tfidf_matched_entity_addr","tfidf_matched_id_addr","tfidf_totalScore_addr"])
  print("Exact address macth records = ",len(addr_match_indexlist))
  logger.info("Adddress match step is completed")
elif file_type in ("Fidelity","Schwab"):
  logger.info("Adddress match step is in progress for fidelity")
  #match_test[addr_match_pstl_code]=match_test[addr_match_pstl_code].astype(str)
  match_test[addr_match_pstl_code]=df_test[addr_match_pstl_code]
  address_org=match_test[["allReg"]+addr_match_pstl_code].apply(lambda row: postalcode_to_plansponsor(row['allReg'],str(row[addr_match_pstl_code[0]]),str(row[addr_match_pstl_code[1]]),str(row[addr_match_pstl_code[2]])),axis=1)
  address_org=pd.DataFrame(list(address_org),columns=["tfidf_matched_entity_addr","tfidf_matched_id_addr","tfidf_totalScore_addr"])
  match_test=pd.concat([match_test,address_org],axis=1)
  addr_match_indexlist=match_test.loc[match_test["tfidf_totalScore_addr"]>0,:].reindex(columns=["allReg","Groundtruth","tfidf_matched_entity_addr","tfidf_matched_id_addr","tfidf_totalScore_addr"])
  print("Exact address match records = ",len(addr_match_indexlist))
  logger.info("Adddress match step is completed")
else:
  addr_match_indexlist=list()
#df_test["plan_sponsor"]=df_test["plan_sponsor"].apply(lambda x:str(x).strip().upper())
matchData.drop(['add_full_t', 'add_full'], axis=1, inplace=True)
match_test["Groundtruth"]=df_test["Groundtruth"]


print(df_test.loc[df_test['PARTYID'].notnull(),:].shape)
print(df_test.loc[df_test['PARTYID'].isnull(),:].shape)


match_test["rowindex"]=match_test.index
entity_cols=[x for x in list(match_test.columns) if "tfidf_matched_entity" in x]+["allReg","rowindex"]
score_cols=[x for x in list(match_test.columns) if "tfidf_totalScore" in x]+["allReg","rowindex"]
id_cols=[x for x in list(match_test.columns) if "tfidf_matched_id" in x]+["allReg","rowindex"]

match_ent=match_test[entity_cols].copy()
match_ent=pd.melt(match_ent,id_vars=["allReg","rowindex"],value_name="match_entity")

match_score=match_test[score_cols].copy()
match_score=pd.melt(match_score,id_vars=["allReg","rowindex"],value_name="match_score")

match_id=match_test[id_cols].copy()
match_id=pd.melt(match_id,id_vars=["allReg","rowindex"],value_name="match_id")

match_predict=pd.concat([match_ent,match_score[["match_score"]],match_id[["match_id"]]],axis=1).sort_values(["rowindex","match_score"],ascending=[True,False]).reset_index(drop=True)
match_predict["match_id"]=match_predict["match_id"].apply(lambda x: ct(x))
match_predict.match_id=match_predict.match_id.astype("Int64")
###match_predict.match_id=match_predict.match_id.astype(str)

match_predict.drop_duplicates(["rowindex","match_id"],inplace=True)
match_predict.reset_index(drop=True,inplace=True)


logger.info("Picking top 5 entities for model step started")
match_predict["fuzz_score"]=match_predict.apply(lambda row:fuzz.token_set_ratio(row['match_entity'],row["allReg"]),axis=1)
match_predict=match_predict.sort_values(["rowindex","fuzz_score"],ascending=[True,False]).reset_index(drop=True)
match_predict.drop_duplicates(["rowindex","match_entity","match_id"],inplace=True)
match_predict.reset_index(drop=True,inplace=True)

if file_type=="DST":
  match_predict=match_predict.groupby(['rowindex']).apply(lambda x: x.nlargest(5,['fuzz_score'])).reset_index(drop=True)
elif file_type=="RPO":
  match_predict=match_predict.groupby(['rowindex']).apply(lambda x: x.nlargest(3,['fuzz_score'])).reset_index(drop=True)
else:
  match_predict=match_predict.groupby(['rowindex']).apply(lambda x: x.nlargest(5,['fuzz_score'])).reset_index(drop=True)
  
match_test = match_test.loc[:,~match_test.columns.duplicated()]


match_test_2=match_test[['Reg_concatanate','Groundtruth', 'rowindex']].copy()
match_plan_data=pd.merge(match_predict,match_test_2,on="rowindex",how="inner")
logger.info("Picking top 5 entities for model step completed")

print(match_plan_data.shape)

sentences1 = [str(x) for x in match_predict['allReg']]
sentences2 = [str(x) for x in match_predict['match_entity']]


#Compute embedding for both lists
print("model prediction started...")
results=[]
count=0
for a,b in zip(sentences1,sentences2):
  count=count+1
  if count%1000==0:
    print(count)
  embeddings1 = model.encode(a, convert_to_tensor=True)
  embeddings2 = model.encode(b, convert_to_tensor=True)
  cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
  results.append((a,b,round(cosine_score.item(),3)))
print("model prediction completed...")


predictions = pd.DataFrame(results, columns=['allReg', 'match_entity', 'score'])
predictions["Groundtruth"]=match_plan_data["Groundtruth"].apply(lambda x:str(x).upper())
predictions["rowindex"]=match_plan_data["rowindex"]
predictions["match_id"]=match_plan_data["match_id"]
predictions.sort_values(["rowindex","score"],ascending=[True,False],inplace=True)


predictions.reset_index(drop=True,inplace=True)
predictions_top2=predictions.groupby("rowindex").head(2)
predictions_top2.reset_index(drop=True,inplace=True)


predictions_top2["top"]=predictions_top2.groupby("rowindex")["score"].rank("first", ascending=False)
predictions_top2["top"]=predictions_top2["top"].apply(lambda x:"SiameseMatch_"+str(int(x)))


predictions_top2['matched_planspnsr'] = predictions_top2.apply(lambda x:partyid_to_plansponsor(x,'match_id'),axis=1)


predictions_spnsr=predictions_top2[["rowindex","top","matched_planspnsr"]]

df1 = predictions_spnsr.groupby(["rowindex","top"])['matched_planspnsr'].aggregate(lambda x: x).unstack().reset_index()
df1.columns=df1.columns.tolist()

predictions_score=predictions_top2[["rowindex","top","score"]]

df2 = predictions_score.groupby(["rowindex","top"])['score'].aggregate(lambda x: x).unstack().reset_index()
df2.columns=["rowindex","SiameseMatch_1_score","SiameseMatch_2_score"]

predictions_id=predictions_top2[["rowindex","top","match_id"]]
predictions_id.match_id=predictions_id.match_id.astype(str)

df3 = predictions_id.groupby(["rowindex","top"])['match_id'].aggregate(lambda x: x).unstack().reset_index()
df3.columns=["rowindex","SiameseMatch_id_1","SiameseMatch_id_2"]

pred=pd.merge(df1,df2,how="inner",on="rowindex")
pred=pd.merge(pred,df3,how="inner",on="rowindex")

logger.info("predictions are calculated and benchmarking step in progress ")

pred=pd.concat([df_test,pred],axis=1)
if "Unnamed: 0" in list(pred.columns):
  pred.drop(columns=["Unnamed: 0","rowindex"],inplace=True)
  
  
pred["Flag_plan"]=pred["Groundtruth"].apply(lambda x:1 if x in plan_sponsor else 0)

chars_to_remove = [")","(",".","|","[","]","{","}","'",",","&","","_","-","/","#","*","`"]
    
def jaccard_similarity(list1 ,list2) :
  try:
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2))/len(s1.union(s2)))
  except Exception as e:
    return 0

def jaccard_match(s1,s2):
  rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
  string1 = re.sub(rx, '', s1)
  string1 = re.sub(' +',' ',string1).upper().strip()
  string2 = re.sub(rx, '', s2)
  string2 = re.sub(' +',' ',string2).upper().strip()
  l1 = string1.split()
  l2 = string2.split()
  return jaccard_similarity(l1,l2)
  
pred["SiameseMatch1_fuzz_score"]=pred.apply(lambda row:fuzz.token_set_ratio(row['SiameseMatch_1'],row["allReg"]),axis=1)
pred["SiameseMatch2_fuzz_score"]=pred.apply(lambda row:fuzz.token_set_ratio(row['SiameseMatch_2'],row["allReg"]),axis=1)

pred["SiameseMatch1_jacc_score"]=pred.apply(lambda row:jaccard_match(str(row['SiameseMatch_1']),str(row["allReg"])),axis=1)
pred["SiameseMatch2_jacc_score"]=pred.apply(lambda row:jaccard_match(str(row['SiameseMatch_2']),str(row["allReg"])),axis=1)

if file_type=="DST":
  swap_index=list(pred.loc[(pred.SiameseMatch2_jacc_score>0.6)&(pred.SiameseMatch2_jacc_score>=pred.SiameseMatch1_jacc_score),:].index)

  list1=['SiameseMatch_1','SiameseMatch_1_score','SiameseMatch_id_1','SiameseMatch1_fuzz_score','SiameseMatch1_jacc_score']
  list2=['SiameseMatch_2','SiameseMatch_2_score','SiameseMatch_id_2','SiameseMatch2_fuzz_score', 'SiameseMatch2_jacc_score']
  for i in swap_index:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp
else:
  swap_index=list(pred.loc[(pred.SiameseMatch1_fuzz_score<pred.SiameseMatch2_fuzz_score)&(pred.SiameseMatch2_jacc_score>pred.SiameseMatch1_jacc_score),:].index)
  list1=['SiameseMatch_1','SiameseMatch_1_score','SiameseMatch_id_1','SiameseMatch1_fuzz_score','SiameseMatch1_jacc_score']
  list2=['SiameseMatch_2','SiameseMatch_2_score','SiameseMatch_id_2','SiameseMatch2_fuzz_score', 'SiameseMatch2_jacc_score']
  for i in swap_index:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp
	
#pred_2.loc[(pred_2.confidence=="")&(pred_2.siamese_acc==1),:].head(125).tail(10)
# CO-COMPANY,BROS-BROTHERS

def jaccard_match_strip(s1,s2):
  string1 = abbr_replace(s1,replacements_strip)
  string1 = re.sub(' +',' ',string1).upper().strip()
  string2 = abbr_replace(s2,replacements_strip)
  string2 = re.sub(' +',' ',string2).upper().strip()
  l1 = string1.split()
  l2 = string2.split()
  return jaccard_similarity(l1,l2)

pred["Match1_jacc_stripscore"]=pred.apply(lambda row:jaccard_match_strip(str(row['SiameseMatch_1']),str(row["allReg"])),axis=1)
pred["Match2_jacc_stripscore"]=pred.apply(lambda row:jaccard_match_strip(str(row['SiameseMatch_2']),str(row["allReg"])),axis=1)


pred = pred.loc[:,~pred.columns.duplicated()]
pred["Reg_concatanate"]=pred["Reg_concatanate"].apply(lambda x:re.sub(' +',' ',str(x)).upper().strip() if len(str(x))>0 else x)
pred["Reg_concatanate1"]=pred["Reg_concatanate"]

pred['allplan_allReg_id'] = pred["Reg_concatanate"].apply(lambda x: allplan_planname_partyid[str(x)] if str(x) in allplan_planname_partyid.keys() else -9999)
pred['allplan_exact_planname'] = pred.apply(lambda x: allplan_party_id_to_org[x["allplan_allReg_id"]] if x['allplan_allReg_id']!=-9999 else "",axis=1)

pred['ml_plan_id'] = pred.apply(lambda x: allplan_planname_planid[x['Reg_concatanate']] if (x['allplan_allReg_id']!=-9999)&(x['Reg_concatanate'] in allplan_planname_planid.keys()) else np.nan,axis=1)
pred['ml_plan_name'] = pred.apply(lambda x:allplan_planid_planname[x['ml_plan_id']] if (x['ml_plan_id']!=-9999)&(pd.notna(x['ml_plan_id'])) else None,axis=1)


print("Exact plan matching records = {} ".format(str(pred[pred['allplan_allReg_id']!=-9999].shape[0])))
print("Exact plan matching records = {} ".format(str(pred[pred['ml_plan_id'].notnull()].shape[0])))

swap_index_partyid_2=list(pred.loc[(pred.allplan_allReg_id!=-9999)&(pred.allplan_allReg_id!=pred.SiameseMatch_id_1)&(pred.allplan_allReg_id==pred.SiameseMatch_id_2),:].index)
list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
print(len(swap_index_partyid_2))
for i in swap_index_partyid_2:
  for col1,col2 in zip(list1,list2):
    temp=pred.loc[i,col1]
    pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,col2]=temp
    
swap_index_3=list(pred.loc[(pred.allplan_allReg_id!=-9999)&(pred.allplan_allReg_id!=pred.SiameseMatch_id_2)&(pred.allplan_allReg_id!=pred.SiameseMatch_id_1),:].index)

list1=['SiameseMatch_2','SiameseMatch_id_2','SiameseMatch_1','SiameseMatch_id_1']
list2=['SiameseMatch_1','SiameseMatch_id_1','allplan_exact_planname','allplan_allReg_id']

print(len(swap_index_3))
for i in swap_index_3:
    for col1,col2 in zip(list1,list2):
      pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,"SiameseMatch_2_score"]=pred.loc[i,"SiameseMatch_1_score"]
    embeddings1 = model.encode(pred.loc[i,"SiameseMatch_1"], convert_to_tensor=True)
    embeddings2 = model.encode(pred.loc[i,"allplan_exact_planname"], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    pred.loc[i,"SiameseMatch_1_score"]=round(cosine_score.item(),3) 
	
	
chars_to_remove = [")","(",".","|","[","]","{","}","'",",","&","","_","-","/","#","*","`",",",":",";"]

def planname_process(raw_reg):
  try:
    if (len(str(raw_reg))>0)&(pd.notna(raw_reg)):
      rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
      pr_reg = re.sub(rx, ' ', raw_reg)
      pr_reg = re.sub(' +',' ',pr_reg).upper().strip()   
      return pr_reg
    else:
      return raw_reg
  except Exception as e:
    print(e)
    return raw_reg 
	
#code added by Ganapathy - To extract plan type from raw reglines
def reg_con_process(row1):
  text1=fetch_plan_type(str(row1['Reg_concatanate1']).strip().upper())
  text1=clean(text1)
  text2=fetch_plan_type(str(row1['Reg_concatanate_process']).strip().upper())
  if (text2=="") & (text1!=""):
    text3=row1["allRegc"]+" "+text1.strip().upper()    
    text4=planname_process(str(text3))
    if (len(str(text4))==0):
      text4=text3
      
    return [text3,text4]
  else:
    return [row1['Reg_concatanate'],row1['Reg_concatanate_process']]
  
  
all_plans_org["plan_name_proces"]=all_plans_org["plan_name"].apply(lambda x:planname_process(str(x)) if len(str(x))>0 else x)
if file_type=="DST":
  pred["Reg_concatanate"]=pred["allRegc"]
  pred["Reg_concatanate_process"]=pred["allRegc"].apply(lambda x:planname_process(str(x)) if len(str(x))>0 else x)
  
  temp_pred=pred.apply(lambda row: reg_con_process(row),axis=1)
  temp_pred=pd.DataFrame(list(temp_pred),columns=["Reg_concatanate","Reg_concatanate_process"])
  pred.drop(["Reg_concatanate","Reg_concatanate_process"], axis=1, inplace=True)
  pred=pd.concat([pred,temp_pred],axis=1)

  matchData["org_name_process"]=matchData["org_name"].apply(lambda x:planname_process(str(x)) if len(str(x))>0 else x)
else:
  pred["Reg_concatanate_process"]=pred["Reg_concatanate"].apply(lambda x:planname_process(str(x)) if len(str(x))>0 else x)
  
 
if file_type=="DST":
  pred = pred.loc[:,~pred.columns.duplicated()]

  pred['partyid_exact_allReg'] = pred.apply(lambda x: original_org_partyid[x["allReg"]] if (x["allReg"] in original_org_partyid.keys())&(int(x["allplan_allReg_id"])==-9999) else -9999,axis=1)

  swap_index_partyid_2=list(pred.loc[(pred.partyid_exact_allReg!=-9999)&(pred.partyid_exact_allReg==pred.SiameseMatch_id_2),:].index)
  list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
  list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
  for i in swap_index_partyid_2:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp

  swap_index=list(pred.loc[(pred.partyid_exact_allReg!=-9999)&(pred.partyid_exact_allReg!=pred.SiameseMatch_id_2)&(pred.partyid_exact_allReg!=pred.SiameseMatch_id_1),:].index)
  
  list1=['SiameseMatch_2','SiameseMatch_id_2','SiameseMatch_1','SiameseMatch_id_1']
  list2=['SiameseMatch_1','SiameseMatch_id_1','allReg','partyid_exact_allReg']
  
  for i in swap_index:
    for col1,col2 in zip(list1,list2):
      pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,"SiameseMatch_2_score"]=pred.loc[i,"SiameseMatch_1_score"]
    embeddings1 = model.encode(pred.loc[i,"SiameseMatch_1"], convert_to_tensor=True)
    embeddings2 = model.encode(pred.loc[i,"allReg"], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    pred.loc[i,"SiameseMatch_1_score"]=round(cosine_score.item(),3)  
else:
  pred = pred.loc[:,~pred.columns.duplicated()]

  pred['partyid_exact_allReg'] = pred.apply(lambda x: original_org_partyid[x["Reg_concatanate"]] if (x["Reg_concatanate"] in original_org_partyid.keys())&(int(x["allplan_allReg_id"])==-9999) else -9999,axis=1)
  swap_index_partyid_2=list(pred.loc[(pred.partyid_exact_allReg!=-9999)&(pred.partyid_exact_allReg==pred.SiameseMatch_id_2),:].index)
  list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
  list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
  for i in swap_index_partyid_2:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp

  swap_index=list(pred.loc[(pred.partyid_exact_allReg!=-9999)&(pred.partyid_exact_allReg!=pred.SiameseMatch_id_2)&(pred.partyid_exact_allReg!=pred.SiameseMatch_id_1),:].index)

  list1=['SiameseMatch_2','SiameseMatch_id_2','SiameseMatch_1','SiameseMatch_id_1']
  list2=['SiameseMatch_1','SiameseMatch_id_1','Reg_concatanate','partyid_exact_allReg']
  
  for i in swap_index:
    for col1,col2 in zip(list1,list2):
      pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,"SiameseMatch_2_score"]=pred.loc[i,"SiameseMatch_1_score"]
    embeddings1 = model.encode(pred.loc[i,"SiameseMatch_1"], convert_to_tensor=True)
    embeddings2 = model.encode(pred.loc[i,"Reg_concatanate"], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    pred.loc[i,"SiameseMatch_1_score"]=round(cosine_score.item(),3)
	
	
print(len(swap_index),len(swap_index_partyid_2))

all_plans_org["plan_name_proces"]=all_plans_org["plan_name"].apply(lambda x:planname_process(str(x)) if len(str(x))>0 else x)
print(all_plans_org.shape)

def planfuzz_matching_nbrs(dataset,matchData,matchNames,config,metadata,ncores):
    try:
        results_main=pd.DataFrame()
        logger.info("Loading ML Search Models")
        logging.info("Fitting count vectorizers and preparing nearest neighbour model .... ")
        partyid=['party_id']
        #ML 1 - Word count
        if ncores>dataset.shape[0]:
            ncores=dataset.shape[0]
        partyids=matchData[partyid].fillna('-1')
        logger.info("matchNames ={},partyids={}".format(str(len(matchNames)),str(len(partyids))))
        pool = Pool(ncores)
        chunks = splitDataFrameIntoSmaller(dataset, ncores)

        #ML 3 - Word weight model
        logging.info("Applying fitted model=word weight on source data .... ")
        typer = 'planfuzz_'
        funcs = partial(applyChunks3, typer,vectorizer_plan,nbrs_plan,matchNames, partyids)
        results3 = pool.map(funcs, chunks)
        results3 = pd.concat(results3)
        results_main=scoreResults3(results3,typer).reset_index(drop=True)

        results_main = results_main.loc[:,~results_main.columns.duplicated()]
        pool.close()
    except Exception as e:
        if pool:
            pool.close()
        logger.error(traceback.format_exc())
    return results_main
  
def shufflescores(row):
  allReg=str(row["source_record"])
  score=list()
  entity_cols=[(x,row[x]) for x in list(row.index) if "planfuzz_matched_entity" in x]
  id_cols=[(x,row[x]) for x in list(row.index) if "planfuzz_matched_id" in x]
  score_cols=[(x,row[x]) for x in list(row.index) if "planfuzz_totalScore" in x]
  for i in entity_cols:
    score.append(fuzz.token_sort_ratio(allReg,str(i[1])))
  sort_index = np.argsort(score)[::-1]
  score.sort(reverse=True)
  for ind,i in zip(sort_index,list(range(0,len(sort_index)))):
    row[entity_cols[i][0]]=entity_cols[ind][1]
    row[id_cols[i][0]]=id_cols[ind][1]
    row[score_cols[i][0]]=score[i]
  return row
plan_data=pred[["Reg_concatanate_process"]].copy()
plan_data.columns=["allReg"]
pred = pred.loc[:,~pred.columns.duplicated()]

matchPlans = all_plans_org['plan_name_proces']
name='plan_fuzzy'
nbrs_plan,vectorizer_plan=fitNearestTFIDF(matchPlans, name)

config["neighbors"]=10
plan_nbrs=planfuzz_matching_nbrs(plan_data,all_plans_org,matchPlans,config,metadata,ncores=config["ncores"])
plan_nbrs=plan_nbrs.apply(lambda row:shufflescores(row),axis=1)
pred=pd.concat([pred,plan_nbrs[['planfuzz_totalScore0', 'planfuzz_matched_entity0','planfuzz_matched_id0', 'planfuzz_totalScore1','planfuzz_matched_entity1', 'planfuzz_matched_id1']]],axis=1)
pred['allplan_fuzzid'] = pred.apply(lambda x: x["planfuzz_matched_id0"] if (x['planfuzz_totalScore0']>=95)&(x["allplan_allReg_id"]==-9999) & (x['partyid_exact_allReg']==-9999) else -9999,axis=1)
pred['allplan_fuzz_orgname'] = pred.apply(lambda x: allplan_party_id_to_org[x["allplan_fuzzid"]] if x['allplan_fuzzid']!=-9999 else "",axis=1)

# pred['ml_plan_id'] = pred.apply(lambda x: int(list(all_plans_org.loc[all_plans_org['plan_name_proces']==x['planfuzz_matched_entity0'],"plan_id"])[0]) if x['allplan_fuzzid']!=-9999 else x["ml_plan_id"],axis=1)model_log
# pred['ml_plan_name'] = pred.apply(lambda x:x["planfuzz_matched_entity0"] if (x['allplan_fuzzid']!=-9999) else x["ml_plan_name"],axis=1)

pred = pred.loc[:,~pred.columns.duplicated()]


print("Fuzz plan matching records = {} ".format(str(pred[pred['allplan_fuzzid']!=-9999].shape[0])))
print("Fuzz plan matching records = {} ".format(str(pred[pred['ml_plan_id'].notnull()].shape[0])))
print("Total records = {} ".format(str(pred.shape[0])))

print(pred[(pred['allplan_fuzzid']!=-9999)&(pred.allplan_allReg_id==-9999)].shape[0])
if pred[(pred['allplan_fuzzid']!=-9999)&(pred.allplan_allReg_id==-9999)&(pred.partyid_exact_allReg==-9999)].shape[0]>0:
  swap_index_partyid_3=list(pred.loc[(pred['allplan_fuzzid']!=-9999)&(pred.allplan_allReg_id==-9999)&(pred.partyid_exact_allReg==-9999)&(pred.allplan_fuzzid!=pred.SiameseMatch_id_1)&(pred.allplan_fuzzid==pred.SiameseMatch_id_2),:].index)
  list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
  list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
  for i in swap_index_partyid_3:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp

  swap_index_partyid_4=list(pred.loc[(pred['allplan_fuzzid']!=-9999)&(pred.allplan_allReg_id==-9999)&(pred.allplan_fuzzid!=pred.SiameseMatch_id_1)&(pred.allplan_fuzzid!=pred.SiameseMatch_id_2),:].index)
  
  list1=['SiameseMatch_2','SiameseMatch_id_2','SiameseMatch_1','SiameseMatch_id_1']
  list2=['SiameseMatch_1','SiameseMatch_id_1','allplan_fuzz_orgname','allplan_fuzzid']
  
  for i in swap_index_partyid_4:
    for col1,col2 in zip(list1,list2):
      pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,"SiameseMatch_2_score"]=pred.loc[i,"SiameseMatch_1_score"]
    embeddings1 = model.encode(pred.loc[i,"SiameseMatch_1"], convert_to_tensor=True)
    embeddings2 = model.encode(pred.loc[i,"allplan_fuzz_orgname"], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    pred.loc[i,"SiameseMatch_1_score"]=round(cosine_score.item(),3)
  print(len(swap_index_partyid_3),len(swap_index_partyid_4))
  
  
pred['partyid_fuzz_allReg'] = pred.apply(lambda x: cdm_org_to_party_id[str(x["Reg_concatanate"]).strip().upper()] if (x["allplan_allReg_id"]==-9999)&(x["allplan_fuzzid"]==-9999)&(x["partyid_exact_allReg"]==-9999)&(str(x["Reg_concatanate"]).strip().upper() in cdm_org_to_party_id.keys()) else -9999,axis=1)
pred['source_to_fuzz_orgname'] = pred.apply(lambda x: cdm_party_id_to_org[x["partyid_fuzz_allReg"]] if (x["partyid_fuzz_allReg"]!=-9999)&(x["partyid_fuzz_allReg"] in cdm_party_id_to_org.keys()) else "",axis=1)
print(pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0])

pred['partyid_fuzz_allReg'] = pred.apply(lambda x: allplan_process_org_partyid[str(x["Reg_concatanate"]).strip().upper()] if (x["partyid_fuzz_allReg"]==-9999)&(x['allplan_allReg_id']==-9999)&(x["allplan_fuzzid"]==-9999)&(x["partyid_exact_allReg"]==-9999)&(str(x["Reg_concatanate"]).strip().upper() in allplan_process_org_partyid.keys()) else -9999,axis=1)
pred['source_to_fuzz_orgname'] = pred.apply(lambda x: allplan_party_id_to_org[x["partyid_fuzz_allReg"]] if (x["partyid_fuzz_allReg"]!=-9999)&(x['source_to_fuzz_orgname']=="")&(x["partyid_fuzz_allReg"] in allplan_party_id_to_org.keys()) else "",axis=1)
print(pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0])

pred['partyid_fuzz_allReg'] = pred.apply(lambda x: cdm_org_to_party_id[str(x["allReg"]).strip().upper()] if (x["allplan_allReg_id"]==-9999)&(x["allplan_fuzzid"]==-9999)&(x["partyid_exact_allReg"]==-9999)&(str(x["allReg"]).strip().upper() in cdm_org_to_party_id.keys()) else -9999,axis=1)
pred['source_to_fuzz_orgname'] = pred.apply(lambda x: cdm_party_id_to_org[x["partyid_fuzz_allReg"]] if (x["partyid_fuzz_allReg"]!=-9999)&(x['source_to_fuzz_orgname']=="")&(x["partyid_fuzz_allReg"] in cdm_party_id_to_org.keys()) else "",axis=1)
print(pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0])

pred['partyid_fuzz_allReg'] = pred.apply(lambda x: allplan_process_org_partyid[str(x["allReg"]).strip().upper()] if (x["partyid_fuzz_allReg"]==-9999)&(x['allplan_allReg_id']==-9999)&(x["allplan_fuzzid"]==-9999)&(x["partyid_exact_allReg"]==-9999)&(str(x["allReg"]).strip().upper() in allplan_process_org_partyid.keys()) else -9999,axis=1)
pred['source_to_fuzz_orgname'] = pred.apply(lambda x: allplan_party_id_to_org[x["partyid_fuzz_allReg"]] if (x["partyid_fuzz_allReg"]!=-9999)&(x['source_to_fuzz_orgname']=="")&(x["partyid_fuzz_allReg"] in allplan_party_id_to_org.keys()) else "",axis=1)
print(pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0])


print(pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0])
if pred[(pred['partyid_fuzz_allReg']!=-9999)].shape[0]>0:
  swap_index_partyid_3=list(pred.loc[(pred['partyid_fuzz_allReg']!=-9999)&(pred.partyid_fuzz_allReg!=pred.SiameseMatch_1)&(pred.partyid_fuzz_allReg==pred.SiameseMatch_2),:].index)
  list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
  list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
  for i in swap_index_partyid_3:
    for col1,col2 in zip(list1,list2):
      temp=pred.loc[i,col1]
      pred.loc[i,col1]=pred.loc[i,col2]
      pred.loc[i,col2]=temp

  swap_index_partyid_4=list(pred.loc[(pred['partyid_fuzz_allReg']!=-9999)&(pred.partyid_fuzz_allReg!=pred.SiameseMatch_1)&(pred.partyid_fuzz_allReg!=pred.SiameseMatch_2),:].index)
  
  list1=['SiameseMatch_2','SiameseMatch_id_2','SiameseMatch_1','SiameseMatch_id_1']
  list2=['SiameseMatch_1','SiameseMatch_id_1','source_to_fuzz_orgname','partyid_fuzz_allReg']
    
  for i in swap_index_partyid_4:
    for col1,col2 in zip(list1,list2):
      pred.loc[i,col1]=pred.loc[i,col2]
    pred.loc[i,"SiameseMatch_2_score"]=pred.loc[i,"SiameseMatch_1_score"]
    embeddings1 = model.encode(pred.loc[i,"SiameseMatch_1"], convert_to_tensor=True)
    embeddings2 = model.encode(pred.loc[i,"source_to_fuzz_orgname"], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
    pred.loc[i,"SiameseMatch_1_score"]=round(cosine_score.item(),3)
  print(len(swap_index_partyid_3),len(swap_index_partyid_4))
  
  

if ((file_type=="DST")|(file_type=="Fidelity")|(file_type=="Schwab")):
  pred["confidence"]=""
  if len(addr_match_indexlist)>0:
    for i in list(addr_match_indexlist.index):
      addr_id=addr_match_indexlist.loc[i,"tfidf_matched_id_addr"]
      addr_entity=addr_match_indexlist.loc[i,"tfidf_matched_entity_addr"]
      allReg=addr_match_indexlist.loc[i,"allReg"]
      embeddings1 = model.encode(allReg, convert_to_tensor=True)
      embeddings2 = model.encode(addr_entity, convert_to_tensor=True)
      cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
      if pred.loc[(pred.index==i)&(pred.SiameseMatch_id_1!=addr_id)&(pred.SiameseMatch_id_2!=addr_id)&(pred["allplan_allReg_id"]==-9999)&(pred["partyid_exact_allReg"]==-9999),:].shape[0]>0:        
        pred.loc[(pred.index==i),'confidence']="AddressMatch"
        pred.loc[(pred.index==i),'SiameseMatch_2']=pred.loc[(pred.index==i),'SiameseMatch_1']
        pred.loc[(pred.index==i),'SiameseMatch_id_2']=pred.loc[(pred.index==i),'SiameseMatch_id_1']
        pred.loc[(pred.index==i),'SiameseMatch_2_score']=pred.loc[(pred.index==i),'SiameseMatch_1_score']        
        pred.loc[(pred.index==i),'SiameseMatch_1']=addr_entity
        pred.loc[(pred.index==i),'SiameseMatch_id_1']=addr_id
        pred.loc[(pred.index==i),'SiameseMatch_1_score']=round(cosine_score.item(),3)
      elif pred.loc[(pred.index==i)&(pred.SiameseMatch_id_1!=addr_id)&(pred.SiameseMatch_id_2==addr_id)&(pred["allplan_allReg_id"]==-9999)&(pred["partyid_exact_allReg"]==-9999),:].shape[0]>0:
        list1=['SiameseMatch_1','SiameseMatch_id_1',"SiameseMatch_1_score"]
        list2=['SiameseMatch_2','SiameseMatch_id_2',"SiameseMatch_2_score"]
        for col1,col2 in zip(list1,list2):
          temp=pred.loc[i,col1]
          pred.loc[i,col1]=pred.loc[i,col2]
          pred.loc[i,col2]=temp
        pred.loc[(pred.index==i),'confidence']="AddressMatch"
      else:
        pred.loc[(pred.index==i)&((pred.SiameseMatch_id_1==addr_id)&(pred.SiameseMatch_id_2!=addr_id))&(pred["allplan_allReg_id"]==-9999)&(pred["partyid_exact_allReg"]==-9999),'confidence']="AddressMatch"
        
        
        
pred1_1=pred.copy()

from Levenshtein import distance
pred["Levenshtein_score1"]=pred.apply(lambda x:distance(str(x["SiameseMatch_1"]),str(x["allReg"])),axis=1)
pred["Levenshtein_score2"]=pred.apply(lambda x:distance(str(x["SiameseMatch_2"]),str(x["allReg"])),axis=1)

pred["SiameseMatch1_fuzz_score"]=pred.apply(lambda row:fuzz.token_sort_ratio(row['SiameseMatch_1'],row["allReg"]),axis=1)
pred["SiameseMatch2_fuzz_score"]=pred.apply(lambda row:fuzz.token_set_ratio(row['SiameseMatch_2'],row["allReg"]),axis=1)

pred["SiameseMatch1_jacc_score"]=pred.apply(lambda row:jaccard_match(str(row['SiameseMatch_1']),str(row["allReg"])),axis=1)
pred["SiameseMatch2_jacc_score"]=pred.apply(lambda row:jaccard_match(str(row['SiameseMatch_2']),str(row["allReg"])),axis=1)

pred["Match1_jacc_stripscore"]=pred.apply(lambda row:jaccard_match_strip(str(row['SiameseMatch_1']),str(row["allReg"])),axis=1)
pred["Match2_jacc_stripscore"]=pred.apply(lambda row:jaccard_match_strip(str(row['SiameseMatch_2']),str(row["allReg"])),axis=1)


if file_type=="DST":
  pred.loc[(pred["allplan_allReg_id"]!=-9999),"confidence"]="Exact"
  pred.loc[(pred["partyid_exact_allReg"]!=-9999),"confidence"]="Exact"
  pred.loc[((pred["SiameseMatch_1_score"]==1)|(pred["SiameseMatch_2_score"]==1.0))&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["SiameseMatch_1_score"]>=0.9)&(pred["SiameseMatch1_fuzz_score"]>=90)&(pred["SiameseMatch1_jacc_score"]>0.75)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["Match1_jacc_stripscore"]==1)&(pred["SiameseMatch1_fuzz_score"]>=90)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["SiameseMatch1_fuzz_score"]>=95)&(pred["SiameseMatch2_fuzz_score"]>=95)&(pred.partyid_exact_allReg==-9999)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact")&(pred["confidence"]!="Very High"),"confidence"]="High"
  pred.loc[(pred.confidence!="Very High")&(pred["confidence"]!="AddressMatch")&(pred.confidence!="Exact")&(pred["allplan_fuzzid"]!=-9999),"confidence"]="Very High"
  pred.loc[(pred.confidence!="Very High")&(pred["confidence"]!="AddressMatch")&(pred.confidence!="Exact")&(pred["partyid_fuzz_allReg"]!=-9999),"confidence"]="Very High"
  pred.loc[(pred["confidence"]!="Very High")&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="High"
  pred.loc[(pred["SiameseMatch_1_score"]<0.9)&(pred["SiameseMatch1_jacc_score"]<=0.35)&(pred["SiameseMatch2_jacc_score"]<=0.35)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact")&(pred["confidence"]!="Very High"),"confidence"]="Manual inspection"
  pred.loc[(pred.SiameseMatch1_jacc_score==0)&(pred.SiameseMatch2_jacc_score==0),"confidence"]="Manual inspection"
  pred.loc[(pred.confidence=="High")&(pred.Levenshtein_score1<=3)&(pred.SiameseMatch1_fuzz_score>=90)&(pred.SiameseMatch1_jacc_score!=pred.Match1_jacc_stripscore),"confidence"]="Very High"
  pred.loc[(pred.confidence=="High")&(pred.Levenshtein_score1<=1),"confidence"]="Very High"
else:
  pred.loc[(pred["allplan_allReg_id"]!=-9999),"confidence"]="Exact"
  pred.loc[(pred["partyid_exact_allReg"]!=-9999),"confidence"]="Exact"
  pred.loc[((pred["SiameseMatch_1_score"]==1)|(pred["SiameseMatch_2_score"]==1.0))&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["SiameseMatch_1_score"]>=0.95)&(pred["SiameseMatch1_fuzz_score"]>=95)&(pred["SiameseMatch1_jacc_score"]>0.85)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["SiameseMatch1_fuzz_score"]>=95)&(pred["SiameseMatch2_fuzz_score"]>=95)&(pred.partyid_exact_allReg==-9999)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Very High"),"confidence"]="High"
  pred.loc[(pred.confidence!="Very High")&(pred["confidence"]!="AddressMatch")&(pred.confidence!="Exact")&(pred["allplan_fuzzid"]!=-9999),"confidence"]="Very High"
  pred.loc[(pred.confidence!="Very High")&(pred["confidence"]!="AddressMatch")&(pred.confidence!="Exact")&(pred["partyid_fuzz_allReg"]!=-9999),"confidence"]="Very High"
  pred.loc[(pred["Match1_jacc_stripscore"]==1)&(pred["SiameseMatch1_fuzz_score"]>=90)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="Very High"
  pred.loc[(pred["confidence"]!="Very High")&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact"),"confidence"]="High"

  pred.loc[(pred["SiameseMatch_1_score"]<0.9)&(pred["SiameseMatch1_jacc_score"]<=0.35)&(pred["SiameseMatch2_jacc_score"]<=0.35)&(pred["confidence"]!="AddressMatch")&(pred["confidence"]!="Exact")&(pred["confidence"]!="Very High"),"confidence"]="Manual inspection"
  pred.loc[(pred.SiameseMatch1_jacc_score==0)&(pred.SiameseMatch2_jacc_score==0),"confidence"]="Manual inspection"
  pred.loc[(pred.confidence=="High")&(pred.Levenshtein_score1<=3)&(pred.SiameseMatch1_fuzz_score>=90)&(pred.SiameseMatch1_jacc_score!=pred.Match1_jacc_stripscore),"confidence"]="Very High"
  pred.loc[(pred.confidence=="High")&(pred.Levenshtein_score1<=1),"confidence"]="Very High"
  
  
pred["Address_flag"]="Scoring thresholds"
pred.loc[pred.confidence=="AddressMatch","Address_flag"]="Address Match"
pred.loc[(pred.confidence=="Exact")&(pred["allplan_allReg_id"]!=-9999),"Address_flag"]="Exact Match against Plan Name"
pred.loc[(pred.confidence=="Exact")&(pred["partyid_exact_allReg"]!=-9999),"Address_flag"]="Exact Match against Org Name"
pred.loc[(pred.confidence=="Very High")&(pred.allplan_fuzzid!=-9999),"Address_flag"]="Planname fuzz exact match : removed special characters"
pred.loc[(pred.confidence=="Very High")&(pred.partyid_fuzz_allReg!=-9999),"Address_flag"]="Orgname fuzz exact match : removed special characters"


pred_1=pred.copy()
pred_1['confidence'].value_counts()

#change added by Ganapathy:

logger.info("Mark a record as Exact only if plan type is matching, else mark that record as High - Started") 
pred['plan_type']=""
pred['tagged_high_man_insp']=""

indexlist=list(pred.loc[(pred['allplan_allReg_id']!=-9999) | (pred['allplan_fuzzid']!=-9999) | (pred['confidence']=="Very High"),:].index)

temp_dat=list()
temp_dat1=list()
for i in indexlist:    
  if (int(pred.loc[i,'allplan_allReg_id']))!=-9999:
    pred.loc[i,'plan_type'] = fetch_plan_type(pred.loc[i,"Reg_concatanate"])
    print("Entered allplan_allReg_id !=-9999",i)

  elif (int(pred.loc[i,'allplan_fuzzid'])!=-9999) & (pred.loc[i,'confidence']=="Very High"):    
    text2=fetch_plan_type(pred.loc[i,"Reg_concatanate_process"])    
    temp_dat=all_plans_org.loc[(all_plans_org['party_id']==int(pred.loc[i,"allplan_fuzzid"])) & (all_plans_org['plantype'].apply(lambda x: 1 if ((x in text2)| (text2 in x)) else 0))]       
    if (len(temp_dat)>=1) & (text2!=""):
      pred.loc[i,'plan_type']=text2
      pred.loc[i,'confidence']="Exact"
      print("Entered allplan_fuzzid != -9999 and confidence marked as exact",i)
    else:
      pred.loc[i,'plan_type']=""
#       pred.loc[i,'confidence']="High"
      pred.loc[i,'tagged_high_man_insp']='Plan data not available'
      print("Entered allplan_fuzzid != -9999 and confidence marked as High",i)
      
  elif (pred.loc[i,'confidence']=="Very High"):
    text3=fetch_plan_type(pred.loc[i,"Reg_concatanate_process"])    
    temp_dat1=all_plans_org.loc[(all_plans_org['party_id']==int(pred.loc[i,"SiameseMatch_id_1"])) & (all_plans_org['plantype'].apply(lambda x: 1 if ((x in text3)| (text3 in x)) else 0))]       
    if (len(temp_dat1)>=1) & (text3!=""):
      pred.loc[i,'plan_type']=text3
      print("Entered confidence == Very High and retained the confidence",i)
    else:
      pred.loc[i,'plan_type']=""
#       pred.loc[i,'confidence']="High"
      pred.loc[i,'tagged_high_man_insp']='Plan data not available'
      print("Entered confidence == Very High and marked as High",i)
      
  temp_dat=list()
  temp_dat1=list()
    

logger.info("Mark a record as Exact only if plan type is matching, else mark that record as High - completed")       
      
      
      
pred_2=pred.copy()
pred_2['confidence'].value_counts()

#code Added by Ganapathy
matchData_original["add_full"]=matchData_original['postal_code']+" "+matchData_original['city']+" "+matchData_original['state2']
matchData_original["add_full"]=matchData_original["add_full"].str.upper()

matchData_original["add_full_t"]=matchData_original['postal_code'].str[0:5]+" "+matchData_original['city']+matchData_original['state2']
matchData_original["add_full_t"]=matchData_original["add_full_t"].str.upper() 

if file_type=="Fidelity":
  logger.info("Matchid ambiguity calculation in progress") 
  pred['PLAN_POSTAL_CDE']=pred['PLAN_POSTAL_CDE'].astype(str)
  pred['add_full_t']=pred['PLAN_POSTAL_CDE']+" "+pred['PLAN_CITY']+" "+pred['PLAN_STATE']
  pred["add_full_t"]=pred["add_full_t"].str.upper()  
  
elif file_type=="DST":
  logger.info("Matchid ambiguity calculation in progress")   
  pred['add_full_t']=pred['postal_code']+" "+pred['city']+" "+pred['StateName']
  pred["add_full_t"]=pred["add_full_t"].str.upper()  
  
  
#Code Modified by Ganapathy


pred["matchid_ambiguity"]=0
indexlist=list(pred.loc[pred.confidence.isin(["Very High","Exact","AddressMatch"]),:].index)
temp=list()
temp_3=list()
temp_2=list()
temp_2_2=list()
for i in indexlist:
  temp_2=all_plans_org.loc[(all_plans_org.plan_name==pred.loc[i,'Reg_concatanate'])&(pred.loc[i,'allplan_allReg_id']!=-9999),:]
  if (len(temp_2)>1)&(len(list(set(list(temp_2.party_id))))>1):    
    temp_2_2=matchData_original.loc[(matchData_original.partyid.isin(list(temp_2.party_id))) &((matchData_original.add_full_t==str(pred.loc[i,'add_full_t'])) | (matchData_original.add_full==str(pred.loc[i,'add_full_t']))),:]
            
    if (len(temp_2_2)==1):
      pred.loc[i,'SiameseMatch_1']=list(temp_2_2.org_name)[0]
      pred.loc[i,'SiameseMatch_id_1']=list(temp_2_2.partyid)[0]
      print("Entered as there was one match with address - allplan_allReg_id not equal to -9999 =",i)
    else:
      pred.loc[i,'matchid_ambiguity']=1
      pred.loc[i,'confidence']="High"
      print("Entered as there were no or more than one address matches were found - allplan_allReg_id not equal to -9999 = ",i)

  temp=matchData_original.loc[(matchData_original.org_name==pred.loc[i,'SiameseMatch_1'])&(pred.loc[i,'allplan_allReg_id']==-9999)&(pred.loc[i,'allplan_fuzzid']==-9999),:]
  if (len(temp)>1)&(len(list(set(list(temp.partyid))))>1):

    temp_t=temp.loc[(temp.add_full==str(pred.loc[i,'add_full_t'])) | (temp.add_full_t==str(pred.loc[i,'add_full_t']))]

    if len(temp_t)==1:
      pred.loc[i,'SiameseMatch_1']=list(temp_t.org_name)[0]
      pred.loc[i,'SiameseMatch_id_1']=list(temp_t.partyid)[0]
      print("Entered as there was one match with address =",i)
    elif len(temp_t)==0:
      pred.loc[i,'matchid_ambiguity']=1
      pred.loc[i,'confidence']="High"
      print("Entered as there was no match with address = {}".format(str(i)))
    else:
      pred.loc[i,'matchid_ambiguity']=1
      pred.loc[i,'confidence']="High"
      print("Entered as there were more than one mathces with address = {}".format(str(i)))

  temp_3=all_plans_org.loc[(all_plans_org.plan_name_proces==pred.loc[i,'Reg_concatanate_process'])&(pred.loc[i,'allplan_fuzzid']!=-9999),:]
  if (len(temp_3)>1)&(len(list(set(list(temp_3.party_id))))>1):
    pred.loc[i,'matchid_ambiguity']=1
    pred.loc[i,'confidence']="High"
    print("Fuzzy plan index = {}".format(str(i)))  

  temp=list()
  temp_2=list()
  temp_3=list()
  temp_2_2=list()
logger.info("Matchid ambiguity calculation is completed")
pred.drop(['add_full_t'], axis=1, inplace=True)
matchData_original.drop(['add_full_t', 'add_full'], axis=1, inplace=True)


pred_3=pred.copy()
pred_3['confidence'].value_counts()


logger.info("Old Plan Data calculation in progress")
all_plans_org['plan_year']=pd.DatetimeIndex(pd.to_datetime(all_plans_org['plan_value_date'],errors = 'coerce')).year

def plantype_yr_comp(c_plantype,c_yr,comp_text):
  if (((c_plantype in comp_text) & (c_yr >=2015)) | ((comp_text in c_plantype) & (c_yr >=2015))):
    return 1
  else:
    return 0

indexlist=list(pred.loc[(pred['plan_type']!=""),:].index)
temp_1=list()
all_plans_org_t=list()

for i in indexlist:
  text2=fetch_plan_type(pred.loc[i,"Reg_concatanate_process"])   
  all_plans_org_t=all_plans_org.loc[(all_plans_org['party_id']==int(pred.loc[i,"SiameseMatch_id_1"]))]
  if (len(all_plans_org_t)>=1):
    temp_1=all_plans_org_t.loc[(all_plans_org_t['party_id']==int(pred.loc[i,"SiameseMatch_id_1"])) & (all_plans_org_t[['plantype','plan_year']].apply(lambda row: plantype_yr_comp(row['plantype'],row['plan_year'],text2),axis=1))] 
    if (len(temp_1)>=1) & (text2!=""):
      print("Entered as plan type year is >= 2015 and confidence retained as Very High",i)
    else:
      pred.loc[i,'confidence']="High"
      pred.loc[i,'tagged_high_man_insp']='Old plan data'    
      print("Entered as plan type year is <2015",i)
  
  temp_1=list()
  all_plans_org_t=list()
  
logger.info("Old Plan Data calculation is completed")



pred_4=pred.copy()
print(pred_4['confidence'].value_counts())

logger.info("Identifing Hospital pattern in progress")
indexlist=list(pred.loc[pred.confidence.isin(["Very High","Exact","AddressMatch"]),:].index)

for i in indexlist:
  if pred.loc[i,'Reg_concatanate']!="":
    if (any(j.upper() in pred.loc[i,'Reg_concatanate'].upper() for j in health_list)) & ('hospitality'.upper() not in pred.loc[i,'Reg_concatanate'].upper()):
      pred.loc[i,"confidence"]="High"
      pred.loc[i,'tagged_high_man_insp']='Hospital is present in the raw reg line'

logger.info("Identifing Hospital pattern in completed")

pred_6=pred.copy()
print(pred_6['confidence'].value_counts())  
      
      
for key,value in colmapping_dict.items():
  if key in list(pred.columns):
    pred.rename(columns={key:value},inplace=True)
  if key.lower() in list(pred.columns):
    pred.rename(columns={key.lower():value},inplace=True)
    
    
pred_notnull=pred.loc[pred['PARTYID'].notnull(),:].copy()
pred_notnull=pred_notnull.loc[pred_notnull['Groundtruth']!='',:]


def ct(r):
  try:
    if str(r).strip()=="":
      return np.nan
    elif pd.isna(r):
      return r
    else:
      return int(float(str(r).strip()))
  except Exception as e:
    print("value is r={}".format(r))
    print(e)
    
    
pred_notnull["PARTYID"]=pred_notnull["PARTYID"].apply(lambda x: ct(x))


def accuracy_match(row,gt,match1,match2):
    try:
        if (row[gt]==int(row[match1])) or (row[gt]==int(row[match2])):
            return 1
        else:
            return 0
    except Exception:
        return 0
        
        
pred_notnull['siamese_acc'] = pred_notnull.apply(lambda x: accuracy_match(x, 'PARTYID','SiameseMatch_id_1','SiameseMatch_id_1'), axis=1)
pred_notnull['siamese_acc_2'] = pred_notnull.apply(lambda x: accuracy_match(x, 'PARTYID','SiameseMatch_id_1','SiameseMatch_id_2'), axis=1)


try:
  a=pred_notnull.siamese_acc.value_counts()
  accuracy=round((a[1]/(a[1]+a[0]))*100,2)
  print("accuracy is : ",accuracy)
except Exception as e:
  if 1 in list(a.index):
    accuracy=100
  else:
    accuracy=0
  print(pred_notnull.siamese_acc.value_counts())
  
  
  
#Change added by Ganapathy
if flag_model=="Validation":
  if len(pred_notnull)>0:
    print(pd.crosstab(pred_notnull.confidence,pred_notnull.siamese_acc))
    acc_t=pd.crosstab(pred_notnull['confidence'],pred_notnull['siamese_acc'])
    if (0 in list(acc_t.columns)) & (1 in list(acc_t.columns)):
      accuracy_exact=(acc_t.loc['Exact',1]/(acc_t.loc['Exact',0]+acc_t.loc['Exact',1]))*100 if 'Exact' in list(acc_t.index) else 0
      accuracy_address=(acc_t.loc['AddressMatch',1]/(acc_t.loc['AddressMatch',0]+acc_t.loc['AddressMatch',1]))*100 if 'AddressMatch' in list(acc_t.index) else 0
      records_exact=(acc_t.loc['Exact',0]+acc_t.loc['Exact',1]) if 'Exact' in list(acc_t.index) else 0
      records_address_match=(acc_t.loc['AddressMatch',0]+acc_t.loc['AddressMatch',1]) if 'AddressMatch' in list(acc_t.index) else 0  
    elif 0 in list(acc_t.columns):
      accuracy_exact=0
      accuracy_address=0
      records_exact=(acc_t.loc['Exact',0]) if 'Exact' in list(acc_t.index) else 0
      records_address_match=(acc_t.loc['AddressMatch',0]) if 'AddressMatch' in list(acc_t.index) else 0
    elif 1 in list(acc_t.columns):
      accuracy_exact=100
      accuracy_address=100
      records_exact=(acc_t.loc['Exact',1]) if 'Exact' in list(acc_t.index) else 0
      records_address_match=(acc_t.loc['AddressMatch',1]) if 'AddressMatch' in list(acc_t.index) else 0
    else:
      accuracy_exact=np.nan
      accuracy_address=np.nan
      records_exact=np.nan
      records_address_match=np.nan      
  else:
    accuracy_exact=np.nan
    accuracy_address=np.nan
    records_exact=np.nan
    records_address_match=np.nan
    
Total_data_records=pred.shape[0]
Records_with_groundtruth=pred_notnull.shape[0]

print("Exact Match Record:", records_exact)
print("Exact Match Accuracy:",accuracy_exact)
print("Address Match Record:", records_address_match)
print("Address Match Accuracy:",accuracy_address)
print("Total Records:", Total_data_records)
print("Total Records with Ground truth :", Records_with_groundtruth)



if flag_model=="Validation":
  if len(pred_notnull)>0:
    pred_notnull.loc[pred_notnull.confidence=="Exact","confidence"]="Very High"
    pred_notnull.loc[pred_notnull.confidence=="AddressMatch","confidence"]="Very High"
    print(pd.crosstab(pred_notnull.confidence,pred_notnull.siamese_acc))
    acc_t=pd.crosstab(pred_notnull['confidence'],pred_notnull['siamese_acc'])
    
    if (0 in list(acc_t.columns)) & (1 in list(acc_t.columns)):
      accuracy_veryhigh=(acc_t.loc['Very High',1]/(acc_t.loc['Very High',0]+acc_t.loc['Very High',1]))*100 if 'Very High' in list(acc_t.index) else 0
      records_veryhigh=(acc_t.loc['Very High',0]+acc_t.loc['Very High',1]) if 'Very High' in list(acc_t.index) else 0  
      Manual_inspection=(acc_t.loc['Manual inspection',0]+acc_t.loc['Manual inspection',1]) if 'Manual inspection' in list(acc_t.index) else 0
    elif 0 in list(acc_t.columns):
      accuracy_veryhigh=0
      records_veryhigh=(acc_t.loc['Very High',0]) if 'Very High' in list(acc_t.index) else 0
      Manual_inspection=(acc_t.loc['Manual inspection',0]) if 'Manual inspection' in list(acc_t.index) else 0
    else:
      accuracy_veryhigh=100
      records_veryhigh=(acc_t.loc['Very High',1]) if 'Very High' in list(acc_t.index) else 0
      Manual_inspection=(acc_t.loc['Manual inspection',1]) if 'Manual inspection' in list(acc_t.index) else 0
    
    pred.loc[pred.confidence=="Exact","confidence"]="Very High"
    pred.loc[pred.confidence=="AddressMatch","confidence"]="Very High"
    
    try:
      a=pred_notnull.siamese_acc.value_counts()
      accuracy=round((a[1]/(a[1]+a[0]))*100,2)
      print("accuracy is : ",accuracy)
    except Exception as e:
      if 1 in list(a.index):
        accuracy=100
      else:
        accuracy=0
  else:
    pred.loc[pred.confidence=="Exact","confidence"]="Very High"
    pred.loc[pred.confidence=="AddressMatch","confidence"]="Very High"   
    accuracy=np.nan
    a=pred.confidence.value_counts()
    accuracy_veryhigh=np.nan
    accuracy_high=np.nan
    records_veryhigh=a['Very High'] if 'Very High' in list(a.index) else 0
    records_high=a['High'] if 'High' in list(a.index) else 0
    Manual_inspection=a['Manual inspection'] if 'Manual inspection' in list(a.index) else 0
    Records_with_groundtruth=0
else:
  pred.loc[pred.confidence=="Exact","confidence"]="Very High"
  pred.loc[pred.confidence=="AddressMatch","confidence"]="Very High"
  
  
if flag_model=="Validation":
  if len(pred_notnull)>0:
    print(pd.crosstab(pred_notnull.confidence,pred_notnull.siamese_acc_2))
    acc_t=pd.crosstab(pred_notnull['confidence'],pred_notnull['siamese_acc_2'])    
    if (0 in list(acc_t.columns)) & (1 in list(acc_t.columns)):
      accuracy_high=(acc_t.loc['High',1]/(acc_t.loc['High',0]+acc_t.loc['High',1]))*100 if 'High' in list(acc_t.index) else 0
      records_high=(acc_t.loc['High',0]+acc_t.loc['High',1]) if 'High' in list(acc_t.index) else 0  
    elif 0 in list(acc_t.columns):
      accuracy_high=0
      records_high=(acc_t.loc['High',0]) if 'High' in list(acc_t.index) else 0
    else:
      accuracy_high=100
      records_high=(acc_t.loc['High',1]) if 'High' in list(acc_t.index) else 0
      
      
a=pred.cdm_match_type.value_counts()
for i in ['Not Matched', 'Matched - CDM Algo', 'Matched - Manual']:
  if i not in list(a.index):
    a[i]=0
total_records_matched_not_matched,total_records_matched_cdm_algo,total_records_matched_cdm_Manual=a['Not Matched'],a['Matched - CDM Algo'],a['Matched - Manual']



if flag_model=="Validation":
  if len(pred_notnull)>0:
    print(pd.crosstab(pred_notnull.cdm_match_type,pred_notnull.siamese_acc))
    acc_t=pd.crosstab(pred_notnull['cdm_match_type'],pred_notnull['siamese_acc'])    
    if (0 in list(acc_t.columns)) & (1 in list(acc_t.columns)):
      accuracy_cdm_algo=(acc_t.loc['Matched - CDM Algo',1]/(acc_t.loc['Matched - CDM Algo',0]+acc_t.loc['Matched - CDM Algo',1]))*100 if 'Matched - CDM Algo' in list(acc_t.index) else 0
      records_matched_cdm_algo=(acc_t.loc['Matched - CDM Algo',0]+acc_t.loc['Matched - CDM Algo',1]) if 'Matched - CDM Algo' in list(acc_t.index) else 0  

      accuracy_matched_Manual=(acc_t.loc['Matched - Manual',1]/(acc_t.loc['Matched - Manual',0]+acc_t.loc['Matched - Manual',1]))*100 if 'Matched - Manual' in list(acc_t.index) else 0
      records_matched_Manual=(acc_t.loc['Matched - Manual',0]+acc_t.loc['Matched - Manual',1]) if 'Matched - Manual' in list(acc_t.index) else 0  
      
    elif 0 in list(acc_t.columns):
      accuracy_cdm_algo=0
      records_matched_cdm_algo=(acc_t.loc['Matched - CDM Algo',0]) if 'Matched - CDM Algo' in list(acc_t.index) else 0
      accuracy_matched_Manual=0
      records_matched_Manual=(acc_t.loc['Matched - Manual',0]) if 'Matched - Manual' in list(acc_t.index) else 0
    else:
      accuracy_cdm_algo=100
      records_matched_cdm_algo=(acc_t.loc['Matched - CDM Algo',1]) if 'Matched - CDM Algo' in list(acc_t.index) else 0
      accuracy_matched_Manual=100
      records_matched_Manual=(acc_t.loc['Matched - Manual',1]) if 'Matched - Manual' in list(acc_t.index) else 0
      
      
      
Total_very_high_records=pred.loc[pred['confidence']=='Very High'].shape[0]
Not_matched_very_High=pred.loc[(pred['cdm_match_type']=='Not Matched') & (pred['confidence']=='Very High')].shape[0]
Not_matched_High=pred.loc[(pred['cdm_match_type']=='Not Matched') & (pred['confidence']=='High')].shape[0]

print('Total_very_high_records:',Total_very_high_records)
print('Not_matched_very_High:',Not_matched_very_High)
print('Not_matched_High:',Not_matched_High)


print(pd.crosstab(pred['cdm_match_type'],pred['confidence']))
records_matched_cdm_algo_veryhigh=pred.loc[(pred['cdm_match_type']=='Matched - CDM Algo') & (pred['confidence']=='Very High')].shape[0]

accuracy_cdm_algo_veryhigh=round((pred.loc[(pred['cdm_match_type']=='Matched - CDM Algo') & (pred['confidence']=='Very High')].shape[0]/pred.loc[pred['cdm_match_type']=='Matched - CDM Algo'].shape[0])*100,2) if 'Matched - CDM Algo' in list(pred['cdm_match_type'].value_counts().index) else 0 

records_matched_Manual_veryhigh=pred.loc[(pred['cdm_match_type']=='Matched - Manual') & (pred['confidence']=='Very High')].shape[0]

accuracy_matched_Manual_veryhigh=round((pred.loc[(pred['cdm_match_type']=='Matched - Manual') & (pred['confidence']=='Very High')].shape[0]/pred.loc[pred['cdm_match_type']=='Matched - Manual'].shape[0])*100,2) if 'Matched - Manual' in list(pred['cdm_match_type'].value_counts().index) else 0 

print('\nrecords_matched_cdm_algo_veryhigh:',records_matched_cdm_algo_veryhigh)
print('accuracy_cdm_algo_veryhigh:',accuracy_cdm_algo_veryhigh)
print('records_matched_Manual_veryhigh:',records_matched_Manual_veryhigh)
print('accuracy_matched_Manual_veryhigh',accuracy_matched_Manual_veryhigh)



print(pd.crosstab(pred_notnull['confidence'],pred_notnull['siamese_acc']))
veryhigh_mismatch_records = pred_notnull.loc[(pred_notnull['confidence']=='Very High') & (pred_notnull['siamese_acc']==0)]
print('veryhigh_mismatch_records:',veryhigh_mismatch_records.shape[0])


partyid_vh=pred.loc[(pred.confidence=="Very High")&(pred["ml_plan_id"].isnull()),["Reg_concatanate_process","SiameseMatch_id_1","SiameseMatch_1"]].copy()
planid_vh=list()
planname_vh=list()
for plan,party in zip(list(partyid_vh["Reg_concatanate_process"]),list(partyid_vh["SiameseMatch_id_1"])):
  l=list(all_plans_org.loc[all_plans_org["party_id"]==party,"plan_id"])
  m=list(all_plans_org.loc[all_plans_org["party_id"]==party,"plan_name"])
  if len(l)==1:
    planid_vh.append(l[0])
    planname_vh.append(m[0])
  elif len(l)>1:
    score=list()
    for i in m:
      score.append(fuzz.token_sort_ratio(str(plan),planname_process(str(i))))
    sort_index = np.argsort(score)[::-1]
    score.sort(reverse=True)
    if score[0]>75:
      planid_vh.append(l[sort_index[0]])
      planname_vh.append(m[sort_index[0]])
    else:
      planid_vh.append(l[sort_index[0]])
      planname_vh.append(m[sort_index[0]])
  else:
    planid_vh.append(np.nan)
    planname_vh.append(None)
    
    
partyid_vh["planname_vh"]=planname_vh
partyid_vh["planid_vh"]=planid_vh
for ind,planid,planname in zip(list(partyid_vh.index),planid_vh,planname_vh):
  pred.loc[pred.index==ind,"ml_plan_id"]=planid
  pred.loc[pred.index==ind,"ml_plan_name"]=planname


print("file benchmarking completed...")