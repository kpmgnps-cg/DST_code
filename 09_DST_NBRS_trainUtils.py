import pandas as pd   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import gzip
import pickle
import difflib  
import numpy as np
import re,os,json

separatorwords=customEntities["separatorwords"]

def ngrams(string, n=3):
    #string = string.encode("utf-8", errors="ignore").decode() #remove non ascii chars
    ngrams = zip(*[string[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def num_there(s):
    return any(i.isdigit() for i in s)

def has_there(s):
    return any(i=='#' for i in s)

def clean(x):
  try:
    str_pre=str(x).upper().replace(",INC",", INC").replace(",LLC",", LLC").replace(",PC",", PC").replace("INC.","INC. ").replace("LLC.","LLC. ").replace("PC.","PC. ")    
    global separatorwords
    for i in separatorwords:
        if i in x:
          if len(str_pre.split(i))>1:
              str_pre = str_pre.split(i)[1]
  except Exception:
    str_pre=x
  return str_pre

def removeDigits(stringWithNumbers):
    results = ''.join(c if c not in map(str,range(0,10)) else "" for c in stringWithNumbers)
    return results   
 
    
us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT',
'Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA',
'Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO',
'Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND',
'Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC',
'South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV',
'Wisconsin': 'WI','Wyoming': 'WY','Washington, DC': 'DC','Northern Mariana Islands':'MP','Palau': 'PW','Puerto Rico': 'PR','Virgin Islands': 'VI','District of Columbia': 'DC'}

def regColReplace(row,regcols):
    data=str(row["allReg_clean"])
    for i in regcols:
        data=data.replace(row[i],'')
    return data
	
def regColReplace_empty(row,regcols):
    data=str(row["Reg_concatanate"])
    for i in regcols:
        data=data.replace(row[i],'').upper()
    string=re.sub("(\s\d+\s)"," ",data)
    string=re.sub("\s\d+\s"," ",string)
    string=re.sub("\s\d+$"," ",string)
    string=re.sub("^\d+\s"," ",string)
    string=re.sub("\s+"," ",string)
    string=re.sub(r'\b' + 'FBO' + r'\b', '', string)
    return string
	
    
def regexMatcher(dataframe, regexDictionary):
    dataframe['allReg'] =dataframe['allReg'].apply(lambda x: str(x).upper())
    for pattern in regexDictionary.keys():
        dataframe[pattern] =dataframe['allReg'].apply(lambda x: regexDictionary[pattern].findall(str(x)))
        dataframe[pattern] =dataframe[pattern].apply(lambda x: x[0] if len(x) else '')
    return dataframe 
	
def preprocessString(string,replacewords=['PO BOX', 'P O BOX']):
    global customEntities
    #replacewords=replacewords+list(us_state_abbrev.keys())+list(us_state_abbrev.values())
    replacewords=replacewords+customEntities["replacewords"]
    string = str(string).strip().upper() #make lower case
    replacewords=[str(i).upper() for i in replacewords]    
    for i in replacewords:
        string=re.sub(r'\b' + i + r'\b', '', string)
    chars_to_remove = [")","(",".","|","[","]","{","}","'",",","&","","_","-","/","#","*","`"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, ' ', string) #remove the list of chars defined above
    for i in replacewords:
        string=re.sub(r'\b' + i + r'\b', '', string)
    string = re.sub(' +',' ',string).strip()+ " " # get rid of multiple spaces and replace with a single space
    string=re.sub("(\s\d+\s)"," ",string)
    string=re.sub("\s\d+\s"," ",string)
    string=re.sub("\s\d+$"," ",string)
    string=re.sub("^\d+\s"," ",string)
    string=re.sub("\s+"," ",string)
    #string=re.sub("^LLC\s"," ",string)
    #string=re.sub("^INC\s"," ",string)
    #string=re.sub("^PLLC\s"," ",string)
    string=re.sub("^AND\s"," ",string)
    string=re.sub("^CF 403B\s"," ",string)
    string=re.sub("^RTMT PLAN\s"," ",string)
    string=re.sub("^RET PLAN\s"," ",string)
    string= re.sub(' +',' ',string).strip()
    unique_words = dict.fromkeys(string.split())
    string=' '.join(unique_words)    
    string = re.sub(' +',' ',string).strip()
    return string      
  
def preprocessString_RPO(string,replacewords=['PO BOX', 'P O BOX']):
    orig_string=string
    global customEntities
    replacewords=replacewords+customEntities["replacewords"]
    string = str(string).strip().upper() #make lower case
    replacewords=[str(i).upper() for i in replacewords]    
    for i in replacewords:
        string=re.sub(r'\b' + i + r'\b', '', string)
    chars_to_remove = [")","(",".","|","[","]","{","}","'",",","&","","_","-","/","#","*","`"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, ' ', string) #remove the list of chars defined above
    for i in replacewords:
        string=re.sub(r'\b' + i + r'\b', '', string)
    string = re.sub(' +',' ',string).strip()+ " " # get rid of multiple spaces and replace with a single space
    string=re.sub("\s+"," ",string)
    string=re.sub("^CF 403B\s"," ",string)
    string=re.sub("^RTMT PLAN\s"," ",string)
    string=re.sub("^RET PLAN\s"," ",string)
    string= re.sub(' +',' ',string).strip()
    if len(string)>0:
      return string
    else:
      return orig_string