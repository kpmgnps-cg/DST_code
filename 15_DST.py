filename=filename+"_v1"

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
	
	
pred.rename({"SiameseMatch_id_1":"predicted_partyid","SiameseMatch_id_2":"predicted_partyid_2","SiameseMatch_1":"predicted_orgname","SiameseMatch_2":"predicted_orgname_2"},axis=1,inplace=True)
try:
  pred["filename"]=filename
  if "Groundtruth" not in list(pred.columns):
    pred["Groundtruth"]=""
  pred["predicted_partyid"]=pred["predicted_partyid"].apply(lambda x: ct(x))
  pred["predicted_partyid_2"]=pred["predicted_partyid_2"].apply(lambda x: ct(x))
  
  pred.predicted_partyid=pred.predicted_partyid.astype("Int64")
  pred.predicted_partyid_2=pred.predicted_partyid_2.astype("Int64")
  pred.predicted_partyid=pred.predicted_partyid.astype(str)
  pred.predicted_partyid_2=pred.predicted_partyid_2.astype(str)
  pred["ml_plan_id"]=pred["ml_plan_id"].apply(lambda x: ct(x))
  pred.ml_plan_id=pred.ml_plan_id.astype("Int64")
  pred.ml_plan_id=pred.ml_plan_id.astype(str)
except Exception as e:
  print(e)
  
veryhigh_mismatch_records.to_csv(args["output_folder"]+"veryhigh_mismatch_records_"+filename+".csv",index=False)
logger.info("output saved at ={}".format(str(args["output_folder"]+"veryhigh_mismatch_records_"+filename+".csv")))


pred.to_csv(args["output_folder"]+"siamese_"+model_name+"_"+filename+".csv",index=False)
logger.info("output saved at ={} ".format(str(args["output_folder"]+"siamese_"+model_name+"_"+filename+".csv")))


