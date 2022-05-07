import argparse
import os

import pandas as pd
import psycopg2 as pg


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", default='USERNAME', help="Username used to access the MIMIC Database", type=str)
parser.add_argument("-p", "--password", default='PASSWORD', help="User's password for MIMIC Database", type=str)
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimic' user={0} host='mimic' options='--search_path=mimimciii' password={1}".format(pargs.username,pargs.password))

# Path for processed data storage
exportdir = os.path.join(os.getcwd(),'processed_files')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)


query = """
select subject_id, hadm_id, icustay_id,  extract(epoch from charttime) as charttime, itemid
from mimiciii.chartevents
where itemid in (6035, 3333, 938, 941, 942, 4855, 6043, 2929, 225401, 225437, 225444, 225451, 225454, 225814,
  225816, 225817, 225818, 225722, 225723, 225724, 225725, 225726, 225727, 225728, 225729, 225730, 225731,
  225732, 225733, 227726, 70006, 70011, 70012, 70013, 70014, 70016, 70024, 70037, 70041, 225734, 225735,
  225736, 225768, 70055, 70057, 70060, 70063, 70075, 70083, 226131, 80220)
order by subject_id, hadm_id, charttime

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'culture.csv'),index=False,sep='|')


# 2. microbio (Microbiologyevents)
query = """
select subject_id, hadm_id, extract(epoch from charttime) as charttime, extract(epoch from chartdate) as chartdate 
from mimiciii.microbiologyevents
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir, 'microbio.csv'),index=False,sep='|')

select hadm_id, icustay_id, extract(epoch from startdate) as startdate, extract(epoch from enddate) as enddate
from mimiciii.prescriptions
where gsn in ('002542','002543','007371','008873','008877','008879','008880','008935','008941',
  '008942','008943','008944','008983','008984','008990','008991','008992','008995','008996',
  '008998','009043','009046','009065','009066','009136','009137','009162','009164','009165',
  '009171','009182','009189','009213','009214','009218','009219','009221','009226','009227',
  '009235','009242','009263','009273','009284','009298','009299','009310','009322','009323',
  '009326','009327','009339','009346','009351','009354','009362','009394','009395','009396',
  '009509','009510','009511','009544','009585','009591','009592','009630','013023','013645',
  '013723','013724','013725','014182','014500','015979','016368','016373','016408','016931',
  '016932','016949','018636','018637','018766','019283','021187','021205','021735','021871',
  '023372','023989','024095','024194','024668','025080','026721','027252','027465','027470',
  '029325','029927','029928','037042','039551','039806','040819','041798','043350','043879',
  '044143','045131','045132','046771','047797','048077','048262','048266','048292','049835',
  '050442','050443','051932','052050','060365','066295','067471')
order by hadm_id, icustay_id
"""

d = pd.read_sql_query(query,conn)
d.to_csv(os.path.join(exportdir,'abx.csv'),index=False,sep='|')

