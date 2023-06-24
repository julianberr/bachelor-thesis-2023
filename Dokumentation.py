# Packages Used:
import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import math
from multiprocessing import Pool
from multiprocessing import Manager
from xbrl.cache import HttpCache
from xbrl.instance import XbrlParser, XbrlInstance
from dateutil import parser
from functools import cmp_to_key
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
from scipy.stats.mstats import winsorize
import pendulum
import seaborn as sns
from numba import njit
import matplotlib.pyplot as plt

# Soruces used: 
# SEC EDGAR API and the corresponding files on the SEC EDGAR Platform converted using py-xbrl
# Polygon.io Developer Plan
# Twelve Data
# Interest Rate Data https://www.federalreserve.gov/releases/h15/default.htm

# API ENDPOINTS SEC
ep_companytickers = "https://www.sec.gov/files/company_tickers.json" # Get all tickers by cik
ep_submissions = "https://data.sec.gov/submissions/CIK[_CIK_NUMBER_].json" # Get all file submissions and additional information (sic code)
ep_companyfacts = "https://data.sec.gov/api/xbrl/companyfacts/CIK[_CIK_NUMBER_].json" # Get all values for a specific piece of information

# Steps: 

# 0. Set Headers
headers = {'User-Agent': "EMAIL_ADDRESS"} # For identification purposes

# 1. Extract Filings
tickers_cik = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
tickers_cik = pd.json_normalize(pd.json_normalize(tickers_cik.json(), max_level=0).values[0])
tickers_cik["cik_str"] = tickers_cik["cik_str"].astype(str).str.zfill(10)
print(tickers_cik)
data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)

# 2. Extract SIC for each Company using multi processing
time.sleep(0.5)
s = requests.Session()
# Extract Each SIC and list of Ticker symbols using submissions endpoint and multiprocessing (faster calls)
def callSEC(index, row, companies):
    cik = row['cik_str']
    submissions = s.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
    try:
        submissions = submissions.json()
    except KeyError:
        # writeToFIle(companies)
        pass
    try:
        sic = submissions["sic"]
    except KeyError:
        sic = ""
        pass
    try:
        sicdesc = submissions["sicDescription"]
    except KeyError:
        sicdesc = ""
        pass
    print(index)
    ticker = [row["ticker"]]
    if cik in companies:
        currticker = companies[cik]["ticker"]
        print(currticker)
        print(ticker)
        ticker = [*ticker, *currticker]
    companies[cik] = {"title": row["title"], "cik": cik, "sic": sic, "sicdesc": sicdesc, "ticker": ticker}
    # Limit API calls per second
    time.sleep(0.02)
             
if __name__ == '__main__':
    with Manager() as manager:
        A = manager.dict()
        # Only use 2 processors because of the rate limit of the API (10 Calls per second)
        pool = Pool(processes=2)
        pool.starmap(callSEC, [(index, row, A) for index, row in tickers_cik.iterrows()])
        pool.close()
        pool.join()

# 3. Extract only companies with relevant sic code
f = open( "companies.json" , "rb" )
companies = json.load(f)
f.close()
filteredcompanies = {}
filtersic = ["7370","7371", "7372", "7373", "7374"] #SIC Codes that should be selected
for cik in companies:
    content = companies[cik]
    if content['sic'] in filtersic:
        filteredcompanies[cik] = content

# 4. Extract RPO from companyfacts endpoint for us-gaap and ifrs
s = requests.Session()
rpodata = {}
for cik in filteredcompanies:
    concepts = s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers)
    try:    
        concepts = concepts.json()
    except:
        concepts = ""
        pass
    if 'facts' in concepts:
        if 'us-gaap' in concepts['facts']: 
            rpodata[cik] = {'us-gaap': ""}
            try:
                rpodata[cik]['us-gaap'] = concepts['facts']['us-gaap']["RevenueRemainingPerformanceObligation"] # RPO name for us-gaap xbrl
            except KeyError:
                rpodata[cik]['us-gaap'] = "None"
                pass
        elif 'ifrs-full' in concepts['facts']:
            rpodata[cik] = {'ifrs-full': ""}
            try:
                rpodata[cik]['ifrs-full'] = concepts['facts']['ifrs-full']["TransactionPriceAllocatedToRemainingPerformanceObligations"] # RPO name for ifrs xbrl
            except KeyError:
                rpodata[cik]['ifrs-full']  = "None"
                pass
        else:
            rpodata[cik] = {'facts': "None"}
        filteredcompanies[cik]["rpo"] = rpodata[cik] 


# 5. Convert RPO to useable array by cik
f = open( "filteredcrpo.json" , "rb" )
companies = json.load(f)
f.close()
rpovalues = {}
def filterRPO():
    for cik in companies:
        if "rpo" in companies[cik]:
            if "ifrs-full" in companies[cik]["rpo"]:
                if companies[cik]["rpo"]["ifrs-full"] != "None":
                    units = companies[cik]["rpo"]["ifrs-full"]["units"]
                    getUnits(units, cik)
            elif "us-gaap" in companies[cik]["rpo"]:
                if companies[cik]["rpo"]["us-gaap"] != "None":
                    units = companies[cik]["rpo"]["us-gaap"]["units"]
                    getUnits(units, cik)

# Get rid of monetary values
def getUnits(units, cik):
    if "USD" in units:
        rpovalues[cik] = units["USD"]
    elif "EUR" in units:
        rpovalues[cik] = units["EUR"]

# Filter type of Filing
def only10k():
    popciks = []
    for cik in rpovalues:
        filtered = []
        for elem in rpovalues[cik]:
            if elem["form"] == "10-K" or elem["form"] == "20-F":
                filtered.append(elem)
        if len(filtered):
            rpovalues[cik] = filtered
        else: 
            popciks.append(cik)
    for cik in popciks:
        rpovalues.pop(cik, None)
    print(len(rpovalues))
           
# Remove duplicates from later filings (go by end date and get the one that was filed first)
def extractUnique():
    elements = {}
    for cik in rpovalues:
        for value in rpovalues[cik]:
            if cik not in elements:
                elements[cik] = []
            if checkIfExists(elements[cik], "end", value["end"]):  
                exisitngd = datetime.strptime(getElement(elements[cik], "end", value["end"])["filed"],'%Y-%m-%d')
                currd = datetime.strptime(value["filed"],'%Y-%m-%d')
                if exisitngd > currd:
                    for i in elements[cik][getElement(elements[cik], "end", value["end"])]:
                        for key in i:
                            i[key] = value[key];
            else:
                elements[cik].append(value)
    with open('rponoduplicate.json', 'w') as out_file:
        json.dump(dict(elements), out_file, sort_keys = True, indent = 4,
                ensure_ascii = False)

def checkIfExists(array, key, val):
    return val in [el[key] for el in array]

def getElement(array, key, val):
    for el in array:
        if el[key] == val:
            return el
    return False

def getElementID(array, key, val):
    for i, el in array:
        if el[key] == val:
            return i
    return False
        
filterRPO();
only10k();
extractUnique();

# 6. Construct Filing Links for custom xbrl processing due to missing values in API Endpoint companyfacts
def createDocument(object, cik, index):
   starturl = "https://www.sec.gov/Archives/edgar/data/"
   cik = cik
   accessionNumber = object["accessionNumber"][index].replace("-","")
   primarydoc = object["primaryDocument"][index]
   return starturl+cik+"/"+accessionNumber+"/"+primarydoc

for cik in filteredcompanies: 
   concepts = s.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers)
   documents = []
   try:    
      concepts = concepts.json()
   except:
      concepts = ""
      pass
   for index, elem in enumerate(concepts["filings"]["recent"]["form"]):
      if elem == "10-K" or elem == "20-F":
         filingdate = datetime.strptime(concepts["filings"]["recent"]["filingDate"][index],'%Y-%m-%d')
         if filingdate > datetime.strptime("2017-01-01",'%Y-%m-%d'):
            documents.append({"document": createDocument(concepts["filings"]["recent"], cik, index), "filingdate":concepts["filings"]["recent"]["filingDate"][index]})
   filteredcompanies[cik]["filings"] = documents

# 7. use py-xbrl to extract the relevant data from the provided file urls in the previous step
cache: HttpCache = HttpCache('./cache') # Cache already downloaded files
cache.set_headers({'From': 'EMAIL_Address', 'User-Agent': 'py-xbrl/2.1.0'})
parser = XbrlParser(cache)

rpo = {}
for cik in companies:
  for document in reversed(companies[cik]["filings"]):
    schema_url = document["document"]
    try:
      inst: XbrlInstance = parser.parse_instance(schema_url)
      facts = inst.json()
      facts = json.loads(facts)
    except:
      facts = {"facts": {}}
      pass
    for fact in facts["facts"]:
      if "concept" in facts["facts"][fact]["dimensions"]:
        conept = facts["facts"][fact]["dimensions"]["concept"]
        if conept == "RevenueRemainingPerformanceObligation" or conept == "TransactionPriceAllocatedToRemainingPerformanceObligations": # Extract relevant tags similar to previous steps
            if cik not in rpo:
                rpo[cik] = []
            period = facts["facts"][fact]["dimensions"]["period"]
            if not checkIfExists(rpo[cik], "period", period):
               rpo[cik].append({"end": period, "filed": document["filingdate"], "val": facts["facts"][fact]["value"]}) # Add values in same format as the ones received by API Endpoint

# 8. Manually adjust wrong values (thousands missing) and missing values

# 9. Combine API data with manual extraction
f = open( "rponoduplicateadj.json" , "rb" )
apirpo = json.load(f)
f.close()

f = open( "rpocustomextraction.json" , "rb" )
customrpo = json.load(f)
f.close()
for cik in customrpo:
    if cik not in apirpo:
      apirpo[cik] = customrpo[cik]
    else:
      for entry in customrpo[cik]:
         if not checkIfExists(apirpo[cik], "end", entry["end"]):
            apirpo[cik].append(entry)

# 10. Calculate Geometric Mean
def convertToDF():
    dfsource = {}
    for cik in companies:
        rpovalues = []
        for entry in companies[cik]:
            rpovalues.append(entry["val"]);
        dfsource[cik] = rpovalues
    df = pd.DataFrame.from_dict(dfsource, orient='index')
    df = df.transpose()
    getGeometricMEan(df.pct_change())

def getGeometricMEan(dataframe):
    geometricmeans = []
    for col in dataframe:
        totalval = 1
        count = 0
        for value in dataframe[col]:
            if float(value) != 0 and not math.isnan(value):
                totalval = totalval * (float(value)+1)
                count += 1
        if count == 2:
            geometricmean = pow(totalval, 1/count)
            geometricmeans.append(geometricmean)
    getAverage(geometricmeans)

def getAverage(array):
    print(sum(array)/len(array), len(array))

convertToDF()

# 11. Get Tickers available on Data Provider Polygon.io (polygon.py)
startobj = []
count = 0;
requrl ="https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000"
request = s.get(f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000", headers=headers)
request = request.json()
startobj = startobj + request["results"]
count += int(request["count"])
while "next_url" in request and request["next_url"] != "":
    request = s.get(request["next_url"], headers=headers)
    request = request.json()
    startobj = startobj + request["results"]
    count += int(request["count"])
    print(count)

alltickers = {}

for ticker in startobj:
    if "cik" in ticker:
        newticker = []
        stockticker = [ticker["ticker"]]
        if ticker["cik"] in alltickers:
            currticker = alltickers[ticker["cik"]]["ticker"]
            newticker = [*stockticker, *currticker]
            print(newticker)
            ticker["ticker"] = newticker
        else:
            ticker["ticker"] = [ticker["ticker"]]
        alltickers[ticker["cik"]] = ticker

# 12. Extract Ticker specific Data from Polygon Api especially Description (tickerdata.py)
f = open( "./Polygon/polygontickers.json" , "rb" )
tickers = json.load(f)
f.close()

def tickerDetails(cik, data):
    request = s.get(f"https://api.polygon.io/v3/reference/tickers/{tickers[cik]['ticker']}", headers=headers)
    data[cik] = request.json()
    print(len(data), "called")

if __name__ == '__main__':
    with Manager() as manager:
        A = manager.dict()
        pool = Pool(processes=30)
        pool.starmap(tickerDetails, [(cik, A) for cik in tickers])
        pool.close()
        pool.join()
        #writeToFIle(A)
      
# 13. Manually extract relevant Companies with keyword software in their company description by reviewing their description (filternewcomp.py)
f = open("polygoninfo.json" , "rb" )
companies = json.load(f)
f.close()
for cik in companies:
    if "description" in companies[cik]["results"] and "software" in companies[cik]["results"]["description"].lower() and cik not in filteredcompanies:
        print(cik)

# 14. Redo Data adjustments for new Tickers as explained above and manually adjusting data based on pct change irregularities

# 15. Combine relevant Data from previous SIC Code extraction with new Companies

# 16. Extract daily Stock Data for otc and stocks market from 2013 until active date if company is still active (getalldata.py)
def getTickers(market):
  startobj = []
  count = 0;
  request = s.get(f"https://api.polygon.io/v3/reference/tickers?market={market}&active=true&limit=1000", headers=headers)
  request = request.json()
  startobj = startobj + request["results"]
  count += int(request["count"])
  while "next_url" in request and request["next_url"] != "":
      request = s.get(request["next_url"], headers=headers)
      request = request.json()
      startobj = startobj + request["results"]
      count += int(request["count"])
      print(count)
  return startobj

def extractTicker(ticker):
    print(ticker)
    request = s.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2001-01-01/{today}?adjusted=true&sort=asc&limit=50000", headers=headers)
    request = request.json()
    if "results" in request:
      df = pd.DataFrame.from_records(request["results"])
      #print("running")
      df.to_csv(f'Store/{ticker}.csv', encoding='utf-8', index=False)
    else:
      print("no results")

if __name__ == '__main__':
    with Manager() as manager:
        A = manager.dict()
        pool = Pool(processes=30)
        tickers = getTickers("stocks")
        pool.map(extractTicker, [ticker["ticker"] for ticker in tickers])
        pool.close()
        pool.join()

# 17. Get relevant Ticker Symbol for each CIK based on market cap (relevanttickers.py)

f = open( "Polygon/Data/polygontickers.json" , "rb" ) 
alltickers = json.load(f)
f.close()

f = open( "Data/companies.json" , "rb" ) 
companies = json.load(f)
f.close()

f = open( "Data/combinedqrpo.json" , "rb" ) 
relevanttickers = json.load(f)
f.close()

s = requests.Session()
relevant = {}
for cik in relevanttickers:
    if cik in alltickers:
      tickers = alltickers[cik]["ticker"]
    else:
      tickers = companies[cik]["ticker"]
    if len(tickers) > 1:
        tickermc = {}
        for ticker in tickers:
            request = s.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}", headers=headers)
            request = request.json()
            if "market_cap" in request["results"]:
                tickermc[ticker] = int(request["results"]["market_cap"])
            else: 
                tickermc[ticker] = 0;
        highestval = 0
        highestkey = ""
        for ticker in tickermc:
           if int(tickermc[ticker]) > highestval:
              highestkey = ticker
              highestval = int(tickermc[ticker]) 
        if highestval:
           relevant[cik] = [highestkey]
    else:
        relevant[cik] = tickers
            
with open('Polygon/Data/compbyticker.json', 'w') as out_file:
        json.dump(dict(relevant), out_file, sort_keys = True, indent = 4,
                ensure_ascii = False)
        
# 18. Extract Asset Data for all combined entries (assets.py)
f = open("Data/combinedqrpo.json" , "rb" )
companies = json.load(f)
f.close()

def getElementID(array, key, val):
    for i, el in enumerate(array):
        if el[key] == val:
            return i
    return False
s = requests.Session()
for cik in companies:
    concepts = s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers)
    concepts = concepts.json()
    for i, entry in enumerate(companies[cik]):
        asset = 0
        if 'facts' in concepts:
          assets = concepts["facts"]
          if "us-gaap" in assets:
            assets = assets["us-gaap"]["Assets"]["units"]
          else: 
            assets = assets["ifrs-full"]["Assets"]["units"]
          if "USD" in assets:
            assets = assets["USD"]
          elif "EUR" in assets:
            assets = assets["EUR"]
          try:
            asset = int(assets[getElementID(assets, "filed", entry["filed"])]["val"]);
          except:
            asset = 0
            pass
        # if already present in data = manual adjustmen => get manual adjustment
        if "assets" not in companies[cik][i]:
          companies[cik][i]["assets"] = asset
        else:
          asset = companies[cik][i]["assets"]
        companies[cik][i]["aKU"] = int(companies[cik][i]["val"])/asset

with open('Polygon/Data/assetsdata.json', 'w') as out_file:
    json.dump(dict(companies), out_file, sort_keys = True, indent = 4,
            ensure_ascii = False)
# Extract Only Entries with certain distance to eachother

# 19. Convert raw Data to only incorporate wanted data whithout time gaps (extractadj.py)
f = open( "Polygon/Data/assetsdata.json" , "rb" ) 
assets = json.load(f)
f.close()
filteredforms = ["10-K", "20-F", "10-Q"]

def sortfunc(x, y):
  olddate = datetime.strptime(x["end"],'%Y-%m-%d')
  newdate = datetime.strptime(y["end"],'%Y-%m-%d')
  if olddate < newdate:
    return -1
  elif olddate > newdate:
    return 1
  else:
    return 0
  
def adjustfilings(timeframe, data):
    margin = (1000*60*60*24*30)*timeframe
    adjustedoutput = {}
    for cik in data:
      adjustedoutput[cik] = []
      data[cik].sort(key=cmp_to_key(sortfunc))
      for index, element in enumerate(data[cik]):
          if index+1 != len(data[cik]):
            if element["form"] in filteredforms:
              findindex = index+1
              while findindex < len(data[cik]) and data[cik][findindex]["form"] not in filteredforms:
                findindex += 1
              if findindex != len(data[cik]):
                #print(findindex)
                nextelement = data[cik][findindex]
                currdate = int(parser.parse(element["end"]).timestamp()*1000)
                nextdate = int(parser.parse(nextelement["end"]).timestamp()*1000)
                #if "2020-" not in nextelement["end"]:
                if nextdate - currdate < margin and nextelement["val"] != 0 and nextelement["val"] != "null" and element["val"] != 0 and element["val"] != "null" and nextelement["aKU"] != 0 and nextelement["aKU"] != "null" and element["aKU"] != 0 and element["aKU"] != "null":
                  appendobj = {"end": nextelement["end"], "filed": nextelement["filed"], "form": nextelement["form"], "pct_change": float(nextelement["val"])/float(element["val"]), "aku_change": float(nextelement["aKU"])/float(element["aKU"]), "aKU": nextelement["aKU"], "rpo": nextelement["val"]}
                  adjustedoutput[cik].append(appendobj)
    with open('Polygon/Data/adjstfilings.json', 'w') as out_file:
        json.dump(dict(adjustedoutput), out_file, sort_keys = True, indent = 4,
                ensure_ascii = False)
        
adjustfilings(4, assets)

# 20. Convert Index Data to same layout as Stock Data necessary because of different Data provider (Twelve Data) (index.py)
symbol = "SPX"
index = requests.get(f"https://api.twelvedata.com/time_series?symbol={symbol}&outputsize=5000&interval=1day&apikey={apikey}")
index = index.json()

for entry in index["values"]:
    entry["datetime"] = int(parser.parse(entry["datetime"]).timestamp()*1000)

df = pd.DataFrame.from_records(index["values"])
df.rename(columns={"datetime": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}, inplace=True)
df[::-1].to_csv(f'Store/Indeces/{symbol}.csv', encoding='utf-8', index=False)

# 21. Clean Data and Get Dates including Regression Analysis (datacleaning.py)
f = open( "Polygon/Data/compbyticker.json" , "rb" ) 
alltickers = json.load(f)
f.close()

f = open( "Polygon/Data/adjstfilings.json" , "rb" ) 
rpodata = json.load(f)
f.close()

aku = []
akuc = []
rpo = []
rpoabs = []
sret = []
ciks = []
moffset = 1000*60*60*24*30
currindex = "SPX"
indexdata = pd.read_csv(f"Store/Indeces/{currindex}.csv")

def findDates(data, element, current, offset, cik):
    parseddate = int(parser.parse(element["filed"]).timestamp()*1000)+moffset*current
    offsetdate = int(parser.parse(element["filed"]).timestamp()*1000)+moffset*offset
    cdate = findDate(data, parseddate)
    odate = findDate(data, offsetdate)
    if isinstance(cdate, pd.core.series.Series) and isinstance(odate, pd.core.series.Series):
      stockreturn = 0
      indexreturn = 0
      indexreturn = adjustIndex(data, cdate, odate)
      if parseddate > offsetdate:
          stockreturn = (cdate["c"]/odate["c"])-1
      else:
          stockreturn = (odate["c"]/cdate["c"])-1
      return stockreturn - indexreturn
    else:
       return False
    
def findDate(data, date):
    parseddate = date
    indexr = data["t"].searchsorted(parseddate, side="right")
    previndex = indexr - 1
    index = previndex
    if indexr == len(data):
        return False
    else:
        if abs(date - int(data.iloc[indexr]["t"])) < abs(date - int(data.iloc[previndex]["t"])):
            index = indexr
        datediff = abs((data.loc[index, 't']-parseddate)/(1000*60*60*24))
        if datediff > 3:
            parseddate = date
            return False
        else:
            return data.iloc[index]
        
def adjustIndex(data, startdate, enddate):
  cindex = findDate(indexdata, startdate["t"])
  oindex = findDate(indexdata, enddate["t"])
  indexreturn = 0
  if isinstance(cindex, pd.core.series.Series) and isinstance(oindex, pd.core.series.Series):
      if startdate["t"] > enddate["t"]:
          datapos = data.loc[data["t"] == enddate["t"]].index[0]
          indexpos = indexdata.loc[indexdata["t"] == oindex["t"]].index[0]
          indexreturn = (cindex["c"]/oindex["c"])-1
      else:
          datapos = data.loc[data["t"] == startdate["t"]].index[0]
          indexpos = indexdata.loc[indexdata["t"] == cindex["t"]].index[0]
          indexreturn = (oindex["c"]/cindex["c"])-1
      minlen = min(min(len(data["c"][datapos:]), len(indexdata["c"][indexpos:])), 250)
      correlation = np.corrcoef(data["c"][datapos:(datapos+minlen)].to_numpy(), indexdata["c"][indexpos:indexpos+minlen].to_numpy())[0][1]
      indexreturn = correlation*indexreturn
  return indexreturn

for cik in alltickers:
    currentdf = pd.read_csv(f"Store/{alltickers[cik][0]}.csv")
    for element in rpodata[cik]:
      stockreturn = findDates(currentdf, element, 0, -6, cik)
      if stockreturn != False:
        aku.append(element["aKU"])
        akuc.append(element["aku_change"]-1)
        rpo.append(element["pct_change"]-1)
        rpoabs.append(element["rpo"])
        sret.append(stockreturn)
        ciks.append(cik)
    



totalov = pd.DataFrame(list(zip(aku, akuc, rpo, rpoabs, sret, ciks)), columns = ['aKU', 'Assets', 'RPO', 'RPOABS', 'STReturn', 'cik'])
totalov.dropna()
cleandata = {"Assets": 2, "RPO": 2, "aKU": "std"}

# Remove Extreme Outliers
def truncate(datapoint, variant, minv=0, maxv=0):
  if variant == "std":
    # Truncate based on X Sigma
    upper_bound = totalov[datapoint].mean() + totalov[datapoint].std()*minv
    lower_bound = totalov[datapoint].mean() - totalov[datapoint].std()*minv
    totalov.drop(totalov[totalov[datapoint] > upper_bound].index, inplace=True)
    totalov.drop(totalov[totalov[datapoint] < lower_bound].index, inplace=True)
  else:
    totalov.drop(totalov[totalov[datapoint] > maxv].index, inplace=True)
    totalov.drop(totalov[totalov[datapoint] < minv].index, inplace=True)

def standardize(data, elements):
    scaler = StandardScaler()
    newdf = scaler.fit_transform(data[elements])
    data[elements] = pd.DataFrame(newdf, columns = elements)
    return data

def winsorization(datapoint, sigma):
    lower_bound = totalov[datapoint].mean() - totalov[datapoint].std()*sigma
    upper_bound = totalov[datapoint].mean() + totalov[datapoint].std()*sigma
    print(lower_bound, upper_bound)
    lower = percentileofscore(totalov[datapoint], lower_bound)/100
    upper = 1-(percentileofscore(totalov[datapoint], upper_bound)/100)
    # print(totalov[datapoint])
    print(lower, upper)
    print(lower, upper)
    totalov[datapoint] = winsorize(totalov[datapoint], limits=(lower, upper))

def adjustentriecount(treshold):
    alllengths = {}
    lengths = []
    for cik in alltickers:
       if cik in totalov["cik"].values:
          length = len(totalov[totalov["cik"] == cik])
          alllengths[cik] = length
          lengths.append(length)
    lengths = pd.DataFrame(lengths, columns = ['Entries'])
    cutoff = round(float(lengths.max()[0]) * treshold)
    print(f"Minimum entries: {cutoff}")
    for cik in alllengths:
       if float(alllengths[cik]) < cutoff:
          totalov.drop(totalov[totalov["cik"] == cik].index, inplace=True)

def adjustabsoluteval(treshold):
   for cik in alltickers:
      if cik in totalov["cik"].values:
         if totalov[totalov["cik"] == cik]["RPOABS"].mean() < treshold:
            totalov.drop(totalov[totalov["cik"] == cik].index, inplace=True)


truncate("aKU", "std", 5)
truncate("Assets", "pct", -0.7, 1)
truncate("RPO", "pct", -0.7, 1)
totalov = totalov.reset_index()
totalov = totalov.drop('index', axis=1)
totalov = standardize(totalov, ["aKU", "Assets", "RPO", "STReturn"])
winsorization("aKU", 3)
winsorization("Assets", 3)
winsorization("RPO", 3)
adjustentriecount(0.5)
adjustabsoluteval(500000000)

# 22. Create Backtesting Algorithm based on signals
class backtesting:
    def __init__(self, startdate, treshold, strategy):
        print("initiate")
        # Adjust Signal Construction values from 0 to 1
        self.weighting = {"RPO": 0.5, "aKU": 0.5}
        self.startprice = 100
        self.startdate = startdate
        self.currentdate = startdate
        self.treshold = treshold
        self.strategy = strategy
        self.stocks = pd.DataFrame(columns=["startprice", "direction", "return", "flag"])
        # cik (instead of cik => 1), direction (1=long, -1=short, 0=noaction), flag(0=none, 1=flag set), startprice, return
        self.stockvals = np.zeros(shape=(200,5), dtype=np.float64)
        self.stockarrays = {}
        self.signals = pd.DataFrame(columns=["RPO", "aKU", "Assets", "filed"])
        self.portfolio = np.zeros(shape=(len(ciks),2))
        self.portfoliov = {}
        self.parsedcurrdate = ""
        self.trades = {}
        self.backtest()

    def backtest(self):
        print("Progress: 0%", end='', flush=True)
        today = pendulum.now().to_date_string()
        totallen = pendulum.from_format(self.currentdate, 'YYYY-MM-DD').diff(pendulum.now()).in_days()
        count = 0
        global totalov
        # Simulate each day
        while self.currentdate != today:
            newentries = totalov[totalov["filed"] == self.currentdate]
            if len(newentries):
              for index, el in newentries.iterrows():
                  self.signals.loc[el["cik"]] = [el["RPO"],el["aKU"],el["Assets"], el["filed"]]
            self.tradingov()
            currentdate = pendulum.from_format(self.currentdate, 'YYYY-MM-DD').add(days=1)
            self.parsedcurrdate = int(currentdate.timestamp()*1000)
            self.currentdate = currentdate.to_date_string()
            count += 1
            print(f"\rProgress: {round((count/totallen)*100,2)}%", end='', flush=True)
            self.portfoliovalue()
        self.stocks = totalov = pd.DataFrame(self.stockvals, columns = ['cik', 'direction', 'flag', 'startprice', 'return' ])
        self.stocks = self.stocks.set_index("cik")
        self.stocks = self.stocks[self.stocks.index != 0]
        self.sellallstocks()
        print("")
        print("====================")
        print(f"Return: {self.stocks['return'].sum()/(len(self.stocks)*self.startprice)}")
        print("====================")
        self.getGreeks()

    # Get current value of Portfolio
    def portfoliovalue(self):
        results = np.zeros(shape=(200,2), dtype=np.float64)
        for cik in self.stockarrays:
            njitcik = int(f"1{cik}")
            self.portfolio = getportfolioval(self.portfolio, njitcik, self.parsedcurrdate, self.stockarrays[cik])            
        self.portfoliov[self.parsedcurrdate], self.stockvals, results = getcurrentvalue(self.stockvals, self.portfolio, self.startprice, results, self.parsedcurrdate)

    # CAPM Data Output
    def getGreeks(self):
        interest = pd.read_csv(f"Store/Interest/IR.csv")
        self.portfoliov = pd.DataFrame.from_dict(self.portfoliov, orient='index', columns=['c'])
        self.portfoliov = self.portfoliov[self.portfoliov["c"] != 0]
        pd.set_option('mode.chained_assignment', None)
        self.portfoliov.reset_index(inplace=True)
        self.portfoliov["date"] = self.portfoliov["index"].apply(lambda x: pendulum.from_timestamp(x/1000).to_date_string())
        indexd = indexdata
        indexd["date"] = indexd["t"].apply(lambda x: pendulum.from_timestamp(x/1000).to_date_string())
        fig, ax = plt.subplots(figsize=(30, 6))
        self.portfoliov["plotdate"] = self.portfoliov["date"].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d').date() )
        graph = sns.lineplot(data=self.portfoliov, x='plotdate', y='c', ax=ax)
        graph.axhline(y=1, color="red")
        ax.set(xlabel=None)
        ax.set_ylabel('Wert des Portfolios')
        ax.set_title('Analyse des Portfolios (50% RPO - 50% aKU Strategie)')
        ax.grid(True)
        ax.xaxis.set_major_locator(dates.MonthLocator(interval=3))
        self.portfoliov.drop(["plotdate"], axis=1, inplace=True)
        printportfolio = self.portfoliov
        interest = interest[interest["3m"] != "ND"]
        self.portfoliov = self.portfoliov[self.portfoliov['date'].isin(indexd['date']) & self.portfoliov['date'].isin(interest['t'])]
        self.portfoliov["c"] = self.portfoliov["c"].pct_change()
        self.portfoliov = self.portfoliov[self.portfoliov["c"] != 0].reset_index(drop=True)
        indexd = indexd[indexd['date'].isin(self.portfoliov['date'])]
        interest = interest[interest['t'].isin(self.portfoliov['date'])].reset_index(drop=True)
        indexd["c"] = indexd["c"].pct_change() 
        self.portfoliov = self.portfoliov.dropna().reset_index(drop=True)
        indexd = indexd.dropna().reset_index(drop=True)
        interest['3m'] = interest['3m'].astype(float)
        # 3 Month risk free rate converted to daily (trading day adjusted (21 trading days))
        IR = interest["3m"][1:].reset_index(drop=True)
        IR = ((1 + IR/100)**(1/(21*3)) -1)
        # Return Market
        Rm = (indexd["c"] + 1).apply(np.log) - IR
        # Return Portfolio
        self.portfoliov["c"] = (self.portfoliov["c"] + 1).apply(np.log)  - IR
        Rp = self.portfoliov.drop(["index", "date"], axis=1)
        X = sm.add_constant(Rm)
        model = sm.OLS(Rp["c"],X).fit()
        print(model.summary())
        print(f"Alpha: {model.params['const']}, Beta: {model.params.iloc[1]}")
        plt.show()
        
    # On last day of traiding sell all stocks to get current Portfolio value
    def sellallstocks(self):
        for index, el in self.stocks.iterrows():
            cik = str(index)[1:]
            currstockprice = self.getlastprice(str(int(index))[1:])
            stockprice = el["startprice"]
            prevdirection = el["direction"]
            if self.stocks.loc[index]["flag"] != 1:
                actreturn = 1
                if prevdirection == 1:
                    actreturn = (currstockprice / stockprice)
                elif prevdirection == -1:
                    actreturn = ((stockprice - currstockprice) / stockprice)+1
                investment = self.stocks.loc[index]["return"]
                returnamt = investment*actreturn
                if returnamt < 0:
                    returnamt = 0
                self.stocks.loc[index, "return"] = returnamt

    # Administer individual trades
    def tradingov(self):
        if len(self.signals) >= self.treshold:
            ciks, currsignals = self.constructsignals()
            median = np.median(currsignals)
            #std = np.std(currsignals)
            for index, elem in np.ndenumerate(currsignals):
              cik = ciks[index[0]]
              njitcik = int(f"1{cik}")
              stockprice = self.getcurrentstockprice(cik)
              self.stockvals, prevstockprice = assignov(njitcik, self.stockvals, stockprice, self.startprice)
              self.stockvals = updatedirection(elem, njitcik, median, prevstockprice, self.startprice, self.stockvals)
    
    # Construct Signal
    def constructsignals(self):
        ciks = self.signals.index.to_numpy()
        bsignals = np.zeros(len(ciks))
        for typew, weight in self.weighting.items():
            signal = self.signals[typew].to_numpy() * weight
            bsignals += signal

        return ciks, bsignals
    
    # Get Last Price to sell all stocks
    def getlastprice(self, cik):
        ticker = alltickers[cik][0]
        if ticker not in self.stockarrays:
            currentdf = pd.read_csv(f"Store/{ticker}.csv")
            currentdf = currentdf[["t","c"]].to_numpy()
            self.stockarrays[cik] = currentdf
        else:
            currentdf = self.stockarrays[cik]
        return currentdf[-1][1]
    
    # Adjusted for numba
    def getcurrentstockprice(self, cik):
        ticker = alltickers[cik][0]
        if cik not in self.stockarrays:
            currentdf = pd.read_csv(f"Store/{ticker}.csv")
            currentdf = currentdf[["t","c"]].to_numpy()
            self.stockarrays[cik] = currentdf
        else:
            currentdf = self.stockarrays[cik]
        parseddate = int(parser.parse(self.signals.loc[cik]["filed"]).timestamp() * 1000)
        output = getDate(parseddate, currentdf, 3)
        return output
    
# Create Numba Functions to execute as fast as possible (necesarry due to large amount of data)
# Get Return based on direction and initial investment
@njit
def calculatereturn(direction, stockold, stocknew, startprice, investment):
    returnamt = startprice
    if stockold != 0:
      actreturn = 1
      if direction == 1:
        actreturn = (stocknew / stockold)
      elif direction == -1:
        actreturn = ((stockold - stocknew) / stockold)+1
      returnamt = investment*actreturn
      if returnamt < 0: 
        returnamt = 0
    return returnamt

# Get Date of current position (or closest)
@njit
def getDate(parseddate, data, treshold):
  searchdata = data[:, 0]
  index = np.searchsorted(searchdata, parseddate, side="right")
  if index == len(searchdata):
      return -1
  datediff = abs((searchdata[index] - parseddate) / (1000 * 60 * 60 * 24))
  if datediff > treshold:
      return -1
  return data[index, 1]

@njit 
def modify_row(arr, row_index, new_values):
    modified_arr = arr.copy()
    modified_arr[row_index, :] = new_values
    return modified_arr

# Create basic structure
@njit
def assignov(cik, stockvals, stockprice, initialinvest):
  prevstockprice = 0
  updatevalues = [cik,0,0,0.00,0]
  index = 0
  if not np.any(stockvals[:, 0] == cik):
      index = np.where(stockvals[:, 0] == 0)[0][0]
  else:
      index = np.where(stockvals[:, 0] == cik)[0][0]
      updatevalues[1] = stockvals[index][1]
      updatevalues[3] = stockvals[index][3]
      updatevalues[4] = stockvals[index][4]
  prevstockprice = stockvals[index][3]
  if stockprice != -1:
     updatevalues[3] = stockprice
     updatevalues[2] = 0
  else:
     updatevalues[2] = 1
  stockvals = modify_row(stockvals, index, updatevalues)
  return stockvals, prevstockprice

# Change Direction based on Signal (long or short)
@njit
def updatedirection(signal, cik, median, stockprice, startprice, stockvals):
  action = 1
  index = np.where(stockvals[:, 0] == cik)[0][0]
  stockvalsd = stockvals[index]
  updatevalues = [cik, stockvalsd[1], stockvalsd[2], stockvalsd[3], stockvalsd[4]]
  if signal > median:
      action = 1
  elif signal < median:
      action = -1
  if stockvalsd[2] != 1:
      updatevalues[4] = calculatereturn(stockvalsd[1], stockprice, stockvalsd[3], startprice, stockvalsd[4])
      updatevalues[1] = action
      stockvals = modify_row(stockvals, index, updatevalues)
  return stockvals

# Get Current Value of Stock using stock price and investment
@njit
def getcurrentvalue(tradedata, currentdata, startprice, results, date):
  totalvalue = 0
  totallength = len(np.where(currentdata[:, 0] != 0)[0])
  for elem in currentdata:
    cik = elem[0]
    updates = [cik, 0.00]
    if cik != 0:
      indextd = np.where(tradedata[:, 0] == cik)[0][0]
      trades = tradedata[indextd]
      actreturn = calculatereturn(trades[1], trades[3], elem[1], startprice, trades[4])
      if actreturn == 0:
         tradeupdate = [cik, trades[1], trades[2], elem[1], 0]
         tradedata = modify_row(tradedata, indextd, tradeupdate)
      updates[1] = actreturn
      results = modify_row(results, indextd, updates)
      totalvalue += actreturn
  if totallength == 0:
    totallength = 1
  return totalvalue/(totallength*startprice), tradedata, results

# Get current stock price of ticker
@njit 
def getportfolioval(data, cik, date, stockdata):
  index = 0
  updatevalues = [cik,0.00]
  if not np.any(data[:, 0] == cik):
      index = np.where(data[:, 0] == 0)[0][0]
  else:
      index = np.where(data[:, 0] == cik)[0][0]
  currstockprice = getDate(date, stockdata, 1.5)
  if currstockprice != -1:
    updatevalues[1] = currstockprice
    data = modify_row(data, index, updatevalues)
  return data
