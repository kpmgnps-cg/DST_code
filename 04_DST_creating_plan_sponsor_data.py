cdm_party_org_f1_v1_t1=spark.read.parquet("adl://adlseastus2lasr.azuredatalakestore.net/lasr/data/standard/cdm/cdm_party_org_f1_v1_t1/current")
cdm_party_org_f1_v1_t1=cdm_party_org_f1_v1_t1.toPandas()
cdm_party_org_f1_v1_t1=cdm_party_org_f1_v1_t1[['partyid','orgname', 'partytaxid','partystatustypeid','deletedindicator']]
cdm_party_org_f1_v1_t1.rename(columns = {'partyid':'partyid','orgname': 'org_name','partytaxid':'PARTY_TAX_ID', 'partystatustypeid':'party_status_type_id', 'deletedindicator':'HUB_STATE_IND'}, inplace = True)

cdm_party_relationship_f1_v1_t1=spark.read.parquet("adl://adlseastus2lasr.azuredatalakestore.net/lasr/data/standard/cdm/cdm_party_relationship_f1_v1_t1/current")
cdm_party_relationship_f1_v1_t1=cdm_party_relationship_f1_v1_t1.toPandas()
cdm_party_relationship_f1_v1_t1=cdm_party_relationship_f1_v1_t1[['primaryrelationshipindicator','frompartyid','topartyid','partyrelationshipstatusid']]
cdm_party_relationship_f1_v1_t1.rename(columns={'primaryrelationshipindicator':'primary_indicator', 'frompartyid':'FROM_PARTY_ID', 'topartyid':'to_party', 'partyrelationshipstatusid':'status_id'},inplace=True)

cdm_party_address_f1_v1_t1=spark.read.parquet("adl://adlseastus2lasr.azuredatalakestore.net/lasr/data/standard/cdm/cdm_party_address_f1_v1_t1/current")
cdm_party_address_f1_v1_t1=cdm_party_address_f1_v1_t1.toPandas()
cdm_party_address_f1_v1_t1=cdm_party_address_f1_v1_t1[['addresspurposeid','partyid','addressid','streetline1','streetline2','citynm','countrysubdivisionnmalt','postalcode','countrynmalt']]
cdm_party_address_f1_v1_t1.rename(columns={'addresspurposeid':'purpose_id','partyid':'PARTY_ID','addressid':'ADDRESS_ID','streetline1':	'street_line_1','streetline2':	'street_line_2','citynm':'city','countrysubdivisionnmalt':'state2', 'postalcode':'postal_code','countrynmalt':'country2'},inplace=True)

cdm_party_role_f1_v1_t1=spark.read.parquet("adl://adlseastus2lasr.azuredatalakestore.net/lasr/data/standard/cdm/cdm_party_role_f1_v1_t1/current")
cdm_party_role_f1_v1_t1=cdm_party_role_f1_v1_t1.select('partyid','partyroletypeid','partyrolestatusid','deletedindicator')
cdm_party_role_f1_v1_t1=cdm_party_role_f1_v1_t1.toPandas()
cdm_party_role_f1_v1_t1.rename(columns={'partyid':'party_id' ,'partyroletypeid':'role_Type_id', 'partyrolestatusid':'STATUS_ID', 'deletedindicator':'HUB_STATE_IND'},inplace=True)


if (not parallel_run):
  import pandas as pd
  merge_data1=cdm_party_org_f1_v1_t1.copy()
  merge_data1=merge_data1.drop_duplicates(subset=merge_data1.columns, keep='first').copy()
  merge_data2=cdm_party_relationship_f1_v1_t1.copy()
  merge_data2=merge_data2.drop_duplicates(subset=merge_data2.columns, keep='first').copy()
  merge_data3=pd.merge(merge_data1,merge_data2, how='left', left_on=['partyid'], right_on=['to_party'])
  merge_data3=merge_data3.drop_duplicates(subset=merge_data3.columns, keep='first').copy()
    
  merge_data4=merge_data3.loc[(merge_data3['party_status_type_id']==1901) & (merge_data3['partyid'].isin(list(cdm_party_role_f1_v1_t1.loc[(cdm_party_role_f1_v1_t1['role_Type_id']==4108) & (cdm_party_role_f1_v1_t1['HUB_STATE_IND']!='Y') & (cdm_party_role_f1_v1_t1['STATUS_ID']==1801),'party_id']))) & (merge_data3['status_id']==1501.0) & (merge_data3['HUB_STATE_IND']=='N')]
  merge_data4=merge_data4.drop_duplicates(subset=merge_data4.columns, keep='first').copy()
    
  merge_data5=pd.merge(merge_data4,cdm_party_address_f1_v1_t1, how='left', left_on=['FROM_PARTY_ID'], right_on=['PARTY_ID'])
  merge_data5=merge_data5.drop_duplicates(subset=merge_data5.columns, keep='first').copy()
  merge_data5=merge_data5[['partyid',	'org_name',	'PARTY_TAX_ID',	'primary_indicator','purpose_id','street_line_1','street_line_2','city','state2','postal_code','country2']].copy()
  print(merge_data5.shape)


if (not parallel_run):
  merge_data5.to_csv(args["input_folder"]+"cdm_all_plan_sponsors_latest.csv",index=False)
  print("Plan Sponsor data saved at ={}".format(str(args["input_folder"]+"cdm_all_plan_sponsors_latest.csv")))
  
  
