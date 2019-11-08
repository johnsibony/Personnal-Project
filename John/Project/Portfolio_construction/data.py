"""
Importation of the data.
"""

# Author: John Sibony <john.sibony@hotmail.fr>

from util import *
from password import *
import pandas as pd
from sqlalchemy import create_engine

def extraction_data(link_engine, query):
    """Extraction of the data using Vadim's database.
    :param link_engine: Link to extact the data (see password.py file).
    :param query: SQL query."""
    engine = create_engine(link_engine)
    data = pd.read_sql_query(query, engine)
    return data

def import_data(index, contract, start_date='2006-01-01', freq=['EW1', 'EW2', 'EW3', 'EW4', 'EW']):
    """Extraction of specific data.
    :param index: Name of the data index ('SP' or 'VIX'or 'VVIX' or '10Ybond' for respectively SP500 or Vix or Volatility of Vix or 10Year TBond index).
    :param contract: Type of Contract ('call' or 'put' or 'future' or 'spot').
    :param start_date: Begining date of the extracted data. String in the format %YYYY-%mm-%dd.
    :param freq: Only valid for SPX index. List of the frequency of the option maturity.
                 (items should be 'EW1' or 'EW2' or 'EW3' or 'EW4' or 'EW' or 'ES' for respectively every 1st Friday or 2nd Friday or 3rd Friday or 4th Friday or end of the month)"""
    link_engine = get_link_engine()
    if(index=='SP'):
        if(len(freq)>1):
            freq = str(tuple(freq))
        else:
            freq = freq[0]
            freq = """('"""+str(freq)+"""')"""
        if(contract=='call'):
            query = '''select option_expiration, date, underlying, strike, delta, value, std_skew, dte, iv from data_option.cme_es_ivol_rp where date >= '''+"""'"""+start_date+"""'"""+''' and "root.symbol" in '''+freq+''' and sense = 'c' '''
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'option_expiration'], inplace=True)
            data = data.set_index("date")
        elif(contract=='put'):
            query = '''select option_expiration, date, underlying, strike, delta, value, std_skew, dte, iv from data_option.cme_es_ivol_rp where date >= '''+"""'"""+start_date+"""'"""+''' and "root.symbol" in '''+str(freq)+''' and sense = 'p' '''
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'option_expiration'], inplace=True)
            data = data.set_index("date")
        elif(contract=='future'):
            query = '''select date,expiry_date,close from data_future.cme_es where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'expiry_date'], inplace=True)
            data = data.set_index("date")
        elif(contract=='spot'):
            query = '''select date,close from data_ohlc.cboe_spx where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date'], inplace=True)
            data = data.set_index("date")
    elif(index=='VIX'):
        if(contract=='call'):
            query = '''select date,option_expiration,strike,underlying,value,iv,delta,std_skew,dte from data_option.cbot_vx_ivol_rp where date >= '''+"""'"""+start_date+"""'"""+''' and "root.symbol" = 'VIX' and sense = 'c' '''
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'option_expiration'], inplace=True)
            data = data.set_index("date")
        elif(contract=='put'):
            query = '''select date,option_expiration,strike,underlying,value,iv,delta,std_skew,dte from data_option.cbot_vx_ivol_rp where date >= '''+"""'"""+start_date+"""'"""+''' and "root.symbol" = 'VIX' and sense = 'p' '''
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'option_expiration'], inplace=True)
            data = data.set_index("date")
        elif(contract=='future'):
            query = '''select date,expiry_date,close from data_future.cbot_vx where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'expiry_date'], inplace=True)
            data = data.set_index("date")
        elif(contract=='spot'):
            query = '''select date,close from data_ohlc.cbot_vix where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date'], inplace=True)
            data = data.set_index("date")
    elif(index=='VVIX'):
        if(contract=='spot'):
            query = '''select date,close from data_ohlc.cboe_vvix where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date'], inplace=True)
            data = data.set_index("date")
    elif(index=='10Ybond'):
        if(contract=='future'):
            query = '''select date,expiry_date,close from data_future.cme_ty where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date', 'expiry_date'], inplace=True)
            data = data.set_index("date")
            data['underlying'] = 0
        elif(contract=='spot'):
            query = '''select * from data_future_cont.ty1 where date >= '''+"""'"""+start_date+"""'"""
            data =  extraction_data(link_engine, query)
            data.sort_values(['date'], inplace=True)
            data = data.set_index("date")
    try:
        return data
    except:
        raise KeyError('Data not find. Look at the argument allowed in the function import_data in the file data.py')

if __name__ == '__main__':
    import_data('SP', 'spot', '2006-01-01')
