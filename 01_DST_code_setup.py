import mlflow
import mlflow.sklearn

model_log=True

mlflow.set_experiment("/Shared/NAD_DG_DSS/projects/PlansponsorAndFundMatching/PSMAllSources_metrics")

args={}
args["config_folder"]="/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/"
args["input_folder"]="/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/"
args["output_folder"]="/dbfs/FileStore/shared_uploads/kpmprg@capgroup.com/out/"

cuda_device='cuda:2'
adls_url="adl://adlseastus2lasr.azuredatalakestore.net/lasr/sandbox/nad/NKV/"

db_insert_flag=True

#Make this True only when you do not want to pull the Plan sponsor file from the Datalake
parallel_run=False

match_type_value="Manual"

file_type="DST"
flag_model="Validation"

# IMPORTNAT - Check and change filename far below the code to save different verions in output folder and in database
filename="dst_202103.csv"
# New
# dst_202101 dst_202102 dst_202103

# old
# input_reglines_202101
# input_reglines_202102
raw_reglines=["REG_LINE1","REG_LINE2","REG_LINE3","REG_LINE4","REG_LINE5","REG_LINE6"]
# raw_reglines=["REG_LINE1","REG_LINE2","REG_LINE3","REG_LINE4","REG_LINE5"]

colmapping_dict={"REGLINE1":"REG_LINE1",
"REGLINE2":"REG_LINE2",
"REGLINE3":"REG_LINE3",
"REGLINE4":"REG_LINE4",
"REGLINE5":"REG_LINE5",
"REGLINE6":"REG_LINE6",
"TRADING_IDENTIFIER_TYPE_ID":"TRADING_IDENTIFIER_TYPE_ID",
"pt_hub_state":"pt_hub_state",
"PARTYID":"PARTYID",
"ROLE_TYPE_ID":"ROLE_TYPE_ID",
"MATCH_METHOD":"cdm_match_type",
"Reg_concatanate":"raw_source_record",
"TRADE_RECEIVED_DATE":"TRADE_RECEIVED_DATE"
}

# Combination of hospitals/Health care pattern moved from very high category to high category because the data is not clean
health_list=['hospital']


src_cols=[colmapping_dict[i] for i in colmapping_dict.keys()]

database_table_v1="DS_AIMatching_results_DST_ML_FUND_V1"
database_table_v1_columns=["CUM_DSC_NUM","SOCIAL_CDE","REG_LINE1","REG_LINE2","REG_LINE3","REG_LINE4","REG_LINE5","REG_LINE6","TRADING_IDENTIFIER_TYPE_ID","pt_hub_state","PARTYID","Groundtruth","ROLE_TYPE_ID","cdm_match_type","raw_source_record","allReg","city","StateName","postal_code","TRADE_RECEIVED_DATE","predicted_orgname","predicted_partyid","confidence","filename","ml_plan_id","ml_plan_name"]

database_table_v2="DS_AIMatching_results_DST_ML_FUND_V2"
database_table_v2_columns=database_table_v1_columns+["predicted_partyid_2","predicted_orgname_2",'SiameseMatch1_fuzz_score', 'SiameseMatch2_fuzz_score','SiameseMatch1_jacc_score', 'SiameseMatch2_jacc_score','Match1_jacc_stripscore', 'Match2_jacc_stripscore','partyid_exact_allReg',"Address_flag","matchid_ambiguity"]


#code Modified by Ganapathy
picsar_table='DS_PICSAR_COMP_V1'

normalized_table='DS_InstitutionalDataCDMLinkagePartyIDToTradingID_ML_V1'
normalized_table_columns=['PartyID', 'RMPlanID', 'OrgName', 'PartyTypeID', 'OrgIntermediaryNumber', 'ExtPlanID', 'ClientID', 'InvestmentPlatformAccountNumber', 'InvestorAccountNumber', 'CumulativeDiscountNumber', 'SocialCode', 'BrokerIdentificationNumber', 'TradingIdentifierTypeID', 'TradingIDSource', 'MatchIDClean', 'CDM_MATCH_TYPE', 'filename', 'trade_received_date']

norm_colmapping_dict={"predicted_partyid":"PartyID","predicted_orgname":"OrgName",'org_intermediary_number':'OrgIntermediaryNumber',"src_sga_dlr_number":"OrgIntermediaryNumber",'CLIENT_ID':'ClientID',"ext_plan_id":"ExtPlanID","EXT_PLAN_ID":"ExtPlanID","INV_PLTFRM_ACC_NUM":"InvestmentPlatformAccountNumber",'InvestorAccountNumber':'InvestorAccountNumber',"invsr_acct_number":"InvestorAccountNumber",'CumulativeDiscountNumber':'CumulativeDiscountNumber','CUM_DSC_NUM':'CumulativeDiscountNumber',"cdm_match_type":"CDM_MATCH_TYPE","trade_received_date":"trade_received_date", "TRADE_RECEIVED_DATE":"trade_received_date"}


