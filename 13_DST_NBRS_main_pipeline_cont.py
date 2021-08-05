import pyap,usaddress

def addressMap(raw):
  full_address=''
  StateName=''
  postal_code=''
  city= ''
  if pd.notnull(raw):
    addresses=pyap.parse(raw, country='US')
    if len(addresses)>0:
      for address in addresses: 
        address=address.as_dict()
        full_address=address['full_address'] if address['full_address'] else ''
        StateName=address['region1'] if address['region1'] else ''
        postal_code=address['postal_code'] if address['postal_code'] else ''
        city=address['city'] if address['city'] else ''
    else:
      address=usaddress.parse(raw)
      if len(address)>0:
        for item in address:          
          if item[1]!='Recipient':
            full_address=full_address+item[0]+' ' if full_address+item[0] in raw else full_address.strip()+item[0]+" "
            if item[1]=='StateName':
              StateName=StateName+item[0]+' '
            if item[1]=='PlaceName':
              city=city+item[0]+' '
            if item[1]=='ZipCode':
              postal_code=postal_code+item[0]+' '
  return [full_address.strip(),city.strip(),StateName.strip(),postal_code.strip()]
  
  
if __name__ =="__main__":
  try:
    logger.info("Reading source and target data for matching")
    matchData, matchNames=readTarget(config,customEntities,metadata)
    logger.info("Match data shape before duplicate removal ={}".format(str(matchData.shape)))
    
    matchData[metadata['target']['matchColumns']] = matchData[metadata['target']['matchColumns']].apply(lambda x : str(x).upper())
    matchData=matchData.sort_values(["street_line_1","street_line_2","city","state2","postal_code",
                                   "country2"]).drop_duplicates(["partyid","org_name","street_line_1","street_line_2","city","state2","postal_code",
                                                                "country2"])
    matchData['count']=matchData.groupby(["partyid","org_name"])["partyid"].transform('count')
    matchData=matchData.loc[(matchData['count']==1)|((matchData['count']>1)&((pd.notnull(matchData['city']))|(pd.notnull(matchData['state2']))|(pd.notnull(matchData['postal_code']))|(pd.notnull(matchData['country2']))))].reset_index(drop=True)
    plan_sponsor_cdm=[str(x).strip().upper() for x in list(matchData["org_name"])]

    all_plans_org = pd.read_csv(config["allplans_dump"])
    all_plans_org=all_plans_org[all_plans_org.PARTY_STATUS=='Active'].reset_index(drop=True)
#     all_plans_org=all_plans_org.loc[all_plans_org.plan_name.notnull(),:]
    all_plans_org=all_plans_org[["org_name","party_id","party_tax_id"]]
    all_plans_org.rename(columns={"party_id":"partyid"},inplace=True)
    all_plans_org[metadata['target']['matchColumns']] = all_plans_org[metadata['target']['matchColumns']].apply(lambda x : str(x).upper())
    for i in set(list(matchData.columns)).difference(set(list(all_plans_org.columns))):
      all_plans_org[i]=None 

    matchData=pd.concat([all_plans_org,matchData],axis=0).reset_index(drop=True)
    matchData=matchData.sort_values(["street_line_1","street_line_2","city","state2","postal_code",
                                   "country2"]).drop_duplicates(["partyid","org_name","street_line_1","street_line_2","city","state2","postal_code",
                                                                "country2"])
    matchData['count']=matchData.groupby(["partyid","org_name"])["partyid"].transform('count')
    logger.info("Match data shape pre duplicate removal ={}".format(str(matchData.shape)))
    matchData=matchData.loc[(matchData['count']==1)|((matchData['count']>1)&((pd.notnull(matchData['city']))|(pd.notnull(matchData['state2']))|(pd.notnull(matchData['postal_code']))|(pd.notnull(matchData['country2']))))].reset_index(drop=True)

    plan_sponsor=[str(x).strip().upper() for x in list(matchData["org_name"])]
    party_id_list = list(matchData['partyid'])

    matchData_original=matchData.copy()
    matchData[metadata['target']['matchColumns']] = matchData[metadata['target']['matchColumns']].apply(lambda x : normalizer(str(x).upper()))      
    matchNames = matchData[metadata['target']['matchColumns']]
    logger.info("Match data shape post duplicate removal ={}".format(str(matchData.shape)))
      
    if trainSearchmodels(matchData,matchNames):
        logger.info("Fitting ground truth nbrs step is successful")
    print(loadmodels())    

    print(len(plan_sponsor),len(party_id_list))
    original_org_partyid={}
    original_party_id_to_org={}
    for party_id,plan_spnsr in zip(party_id_list,plan_sponsor):
      try :
        original_org_partyid[str(plan_spnsr).upper().strip()] = int(party_id)
        original_party_id_to_org[int(party_id)] = str(plan_spnsr).upper().strip()
      except Exception as e:
        print(e)
  except Exception as e:
    logger.error(traceback.format_exc())
	
	
files=[filename]
if file_type=="PSM":
  metadata["source"]['matchColumns']=['reg_line1','reg_line2','reg_line3','reg_line4',"reg_line5"]
elif (file_type=="RPO"):
  config["neighbors"]=3
  metadata["source"]['matchColumns']=raw_reglines  
elif (file_type=="Schwab"):
  config["neighbors"]=3
  metadata["source"]['matchColumns']=raw_reglines  
elif file_type=="Fidelity":
  config["neighbors"]=10
  metadata["source"]['matchColumns']=raw_reglines
else:
  metadata["source"]['matchColumns']=raw_reglines
  
  
words=["401 K PROFIT SHARING PLAN T","SVGS & RET PLAN","SVGS RET PLAN","SVGS RET",'SERP PLAN','PSP & TRST','RET SAVGS','PSP TRST','RET SAVINGS PLAN',"RETIREMENT PLANS","PROFIT SHARING PLAN TR","401K Plan For","ASSOCIATE BENEFIT PLAN","Employees Retirement Benefit Plan and Trust","Employees Retirement Benefit Plan","Employees Retirement Benefit Trust","Employee Benefit Plan of ","EMPLOYEES BENEFIT PLAN","Target Benefit Plan","Benefit Plan","PROFIT SHARING BENEFIT PLAN","DEFINED BENEFIT PLAN AND TRUST","DEFINED BENEFIT PLAN","DEFINED BENEFIT TRUST","Retirement Benefit Plan","DEFINED BENEFIT PENSION PLAN","DEFINED BENEFIT PENSION PLAN","Benefit Program","SAVINGS BENEFIT PLAN","SALARY SAVINGS 401K PLAN","SALARY SAVINGS PLAN","SALARY SAVINGS","CASH BALANCE PLAN","SECTION 457B","RET PLAN AND TRUST","NAN","Salary Reduction Plan","SLRY SVNGS RET PL","PROFITSHARING","PROFIT SHARING PLN","PROFIT SHARING PLA","PRO FIT","PAI TRUST COMPANY INC","AS AGENT FOR","ADVISOR TRUST INC","ADVISOR TRUST INC","PRFT SHR PLAN","Retirement Savings Plan Trust","SAVINGS PLAN","SAVINGS RETIREMENT PLAN","PRFT SHRNG PL","PROFIT SHARING P","RETIR","401A INCEN","INCEN","RETIREM","RETIR","RETIRE","RETIREMENT PLN TRS","PLN TRS","VOLUNTARY EMPLOYEE SAVIN","ANNUITY PLAN","ANNUITY PLA","ANNUITY P","EMPLOYEE MONEY ACC","ANNUITY","TAX S","EMPLOYEE MONEY","EMPLOYSAVS PROF SHARING TRUST","EMPLOYSAVS","PROF SHARING TRUST","RET PLAN","401K PSP AND TR","RETIREMENTPLAN","DC","401K RET PLAN","PREVAIL WAGE D BACON"]
customEntities['replacewords']=words+customEntities['replacewords']+["401 K",'401K']
metadata['target']['regcols']=["full_address"]



import re
replacements = {r'\bASSOC\b':'ASSOCIATES', 
                r'\bCORP\b':'CORPORATION',
                r'\bINC401K\b':'INC',r'\bINC TR\b':'INC',r'\bPLC401K\b':'PLC',r'\bMGMT\b':'\bMANAGEMENT\b'}

def abbr_replace(text, dic=replacements):
    for i, j in dic.items():
        text = re.sub(i,j,text).upper().strip()
    return text
	
	
try:
  print("processing the file in nbrs matching :"+filename)
  config["source"]=args["input_folder"]+filename
  source=readSource(config,customEntities,metadata)
  source["Reg_concatanate"]=source["allReg"]
  logger.info("Started Preprocessing on training data.... ")
  if file_type=="DST":
    address=source["allReg"].apply(lambda x:addressMap(x))
    address=pd.DataFrame(list(address),columns=["full_address","city","StateName","postal_code"])
    preprocess=pd.concat([source,address],axis=1)
    preprocess['allReg_clean'] = preprocess['allReg'].apply(lambda x: clean(x))
    preprocess["allRegc"]=preprocess.apply(lambda row:regColReplace(row,metadata['target']['regcols']),axis=1)
    preprocess["allReg"]=preprocess["allRegc"].apply(lambda x:preprocessString(x))
    preprocess["allReg"]=preprocess.apply(lambda x:x["allReg"] if len(x["allReg"])>0 else regColReplace_empty(x,metadata['target']['regcols']),axis=1)
  else:
    preprocess=source.copy()
    logger.info("Neareset neighbors = {}".format(str(config["neighbors"])))
    preprocess['allReg_clean'] = preprocess['allReg'].apply(lambda x: clean(x))
    preprocess["allReg"]=preprocess["allReg_clean"].apply(lambda x:preprocessString_RPO(x))
    preprocess["allReg"]=preprocess.apply(lambda x:x["allReg"].replace("BANK","SAVINGS BANK") if "SAVINGS BANK" in x["allReg_clean"] else x["allReg"],axis=1)
    preprocess["allReg"]=preprocess["allReg"].apply(lambda x: abbr_replace(x))
    logger.info("File processing with neighbors")
  logger.info("Completed Preprocessing on training data.... ")

  match_nbrs=matching(preprocess,matchData,matchNames,config,customEntities,metadata,ncores=config["ncores"])
  if file_type=="DST":
    match_nbrs=pd.concat([match_nbrs,preprocess[['allRegc','full_address','city', 'StateName', 'postal_code']],source[["Reg_concatanate"]]],axis=1)
  else:
    match_nbrs=pd.concat([match_nbrs,source[["Reg_concatanate"]]],axis=1)
  reg_nullindex=list(source.loc[(source.Reg_concatanate=="")|(source.Reg_concatanate.isnull()),:].index)
  entity_cols=[x for x in list(match_nbrs.columns) if "tfidf_matched_entity" in x]
  score_cols=[x for x in list(match_nbrs.columns) if "tfidf_totalScore" in x]
  id_cols=[x for x in list(match_nbrs.columns) if "tfidf_matched_id" in x]
  logger.info("Number of empty plan records = {}".format(str(len(reg_nullindex))))
  for i in reg_nullindex:
    for j in entity_cols:
      match_nbrs.loc[i,j]=""
    for k in score_cols:
      match_nbrs.loc[i,k]=0     
    for l in id_cols:
      match_nbrs.loc[i,l]=None     
  match_nbrs.to_csv(args["output_folder"]+"aimatch_"+filename,index=False)
  logger.info("saved preprocessed file at :"+args["output_folder"]+"aimatch_"+filename)
except Exception as e:
  logger.error(traceback.format_exc())
  
  
filename=filename.replace(".csv","")
print(filename)
  
  
