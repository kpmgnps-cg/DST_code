import json
import pandas as pd
import os
import gzip,pickle 
import traceback
import warnings
import datetime
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT',
'Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA',
'Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO',
'Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND',
'Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC',
'South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV',
'Wisconsin': 'WI','Wyoming': 'WY','Washington, DC': 'DC','Northern Mariana Islands':'MP','Palau': 'PW','Puerto Rico': 'PR','Virgin Islands': 'VI','District of Columbia': 'DC'}

def convert(x):
    try:
        x = int(x)
    except Exception as e:
        try:
            x = float(x)
        except Exception as e:
            x = x
    return x
    
def trainSearchmodels(matchData,matchNames):     
    try:
        #logger.info("Applying fitted model=count on source data .... ")
        #nbrs,vectorizer=fitNearestN(matchNames)

        #ML 2 - NGRAM model
        #logger.info("Applying fitted model = ngrams on source data .... ")
        #nbrs,vectorizer=fitNearestGrams(matchNames)
        
        #ML 3 - Word weight model
        logger.info("Applying fitted model = tfidf on source data .... ")      
        name = ''
        nbrs,vectorizer=fitNearestTFIDF(matchNames, name)
        #nbrs,vectorizer=fitNearestTFIDFJaccard(matchNames, name)
        return True
    except Exception:
        logger.error(traceback.format_exc())    
        return False