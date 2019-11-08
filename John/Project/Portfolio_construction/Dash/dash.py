import sys
sys.path.append('..')
from portfolio import *
from regime import *
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
import dash_table
import time
import random
import base64, io
import re

columns = ['Strategy nᵒ', 'Instrument nᵒ', 'Index', 'Contract', 'Position', 'Entry Type', 'End', 'Maturity', 'Delta']
features = pd.DataFrame([['']*len(columns)], columns=columns)
adjustment = pd.DataFrame([['']*2], columns=['Id', 'Function'])
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def upload_data(strategy_id, instrument_id):
    return html.Div([ dash_table.DataTable( id='table_strategies_upload'+str(strategy_id)+str(instrument_id),
                                            data=pd.DataFrame([['Strategy nᵒ'+str(strategy_id), 'Instrument nᵒ'+str(instrument_id)]], columns=['1', '2']).to_dict('records'),
                                            columns=[{'id': '1', 'name': ''}, {'id': '2', 'name': ''}],
                                            style_cell={'textAlign': 'left', 'height':'40px'},
                                            style_header={'height':'0px'},
                                          ),
                      dcc.Upload(id='entry'+str(strategy_id)+str(instrument_id), children=html.Div(['Entry (see template file for format):',html.A("""Select File""")]),style={'height': '43px', 'lineHeight': '43px','borderWidth': '1px', 'borderStyle': 'dashed','borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'}),
                      dcc.Upload(id='weight'+str(strategy_id)+str(instrument_id), children=html.Div(['Weight (see template file for format):',html.A("""Select File""")]),style={'height': '43px', 'lineHeight': '43px','borderWidth': '1px', 'borderStyle': 'dashed','borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'})   
                    ], style={'columnCount': 3})

strategy_layout = html.Div([ html.H1(),
                             html.H4(children='Strategies', style={'color': '#0073aa'}),
                             html.Div(id='weights'),
                             dash_table.DataTable(id='table_strategies',
                                                  data=features.to_dict('records'),
                                                  columns=[{'id': 'Strategy nᵒ', 'name': 'Strategy nᵒ'}] +
                                                          [{'id': 'Instrument nᵒ', 'name': 'Instrument nᵒ'}] +
                                                          [{'id': i, 'name': i, 'presentation': 'dropdown'} for i in features.columns[2:7]] +
                                                          [{'id': i, 'name': i, 'type': 'numeric'} for i in features.columns[7:]],
                                                  dropdown={'Index': {'options': [{'label': 'S&P', 'value': 'S&P'}, {'label': 'Vix', 'value': 'Vix'}, {'label': '10Ybond', 'value': '10Ybond'}]},
                                                           'Contract': {'options': [{'label': 'Call', 'value': 'Call'}, {'label': 'Put', 'value': 'Put'}, {'label': 'Future', 'value': 'Future'}]},
                                                           'Entry Type': {'options': [{'label': 'Underlying', 'value': 'underlying'}, {'label': 'Premium', 'value': 'value'}]},
                                                           'Position': {'options': [{'label': 'Long', 'value': 1}, {'label': 'Short', 'value': -1}]},
                                                           'End': {'options': [{'label': 'Roll', 'value': 'Roll'}, {'label': 'Expiry', 'value': 'Expiry'}]}
                                                           },
                                                  style_data_conditional=[{'if':{'column_id': 'Strategy nᵒ', 'filter_query': '{Strategy nᵒ} eq '+str(i)}, 'backgroundColor': 'rgb({}, {}, {})'.format(*random.sample(list(range(135,256)), 3)), 'color': 'black'} for i in range(100)],
                                                  style_cell={'textAlign': 'left'},
                                                  row_deletable=True,
                                                  row_selectable="multi",
                                                  editable=True
                                                ),
                             html.Div([html.Button(id='button_instrument', children='Add Instrument', n_clicks=0)
                                      ]),
                             html.H1(),
                             html.Div(id='div_upload_data'),
                             html.H1(),
                             html.H4(children='Instrument Adjustment', style={'color': '#0073aa'}),
                             html.Div(id='div_adjustment', children=[ dash_table.DataTable( id='adjustment',
                                                                                            data=adjustment.to_dict('records'),
                                                                                            columns=[{'id': i, 'name': i, 'presentation': 'dropdown'} for i in adjustment.columns],
                                                                                            dropdown={'Function': {'options': [{'label': 'Keep Contract', 'value': 'Keep Contract'}, {'label': 'Stop Contract', 'value': 'Stop Contract'}, {'label': 'Stop Gain', 'value': 'Stop Gain'}, {'label': 'Pause', 'value': 'Pause'}]},
                                                                                                     },
                                                                                            style_cell={'textAlign': 'left'},
                                                                                            editable=True
                                                                                          ),
                                                                      html.Button(id='button_adjustment', children='Apply', n_clicks=0)
                                                                    ], style={'columnCount': 2}),
                             dash_table.DataTable(id='table_adjustment',
                                                  data=pd.DataFrame([['Strategy 0 / Instrument 0']+['']*3], columns=['Id', 'Function', 'Argument', 'Dates']).to_dict('records'),
                                                  columns=[{'id': i, 'name': i} for i in ['Id', 'Function', 'Argument', 'Dates']],
                                                  hidden_columns=['Dates'],
                                                  style_cell={'textAlign': 'left'},
                                                  editable=False,
                                                  row_deletable=True,
                                                ),
                             html.H1(),
                             html.H4(children='Run', style={'color': '#0073aa'}),
                             html.Button(id='button_portfolio', children='Run Portfolio'),
                             dcc.ConfirmDialog(id='select_before_id'),
                             dcc.ConfirmDialog(id='confirm'),
                             dcc.ConfirmDialog(id='upload_file_adjustment'),
                             dcc.Textarea(id='current_function_adjustment', style={'display': 'none'}),
                             dcc.Textarea(id='state_function_adjustment', style={'display': 'none'}),
                             html.Div(id='store_adjustment', style={'display': 'none'})
                          ])

trade_layout = html.Div([html.H1(),
                         html.H4(children='Portfolio', style={'color': '#0073aa'}),
                         dash_table.DataTable(id='portfolio_trades'),
                         dcc.Loading(id="loading_table_trades", children=[html.Div(id="output_table_trades")], type="default"),
                         html.H1(),
                         dcc.Loading(id="loading_trade_instrument", children=[html.Div(id="output_trade_instrument")], type="default"),
                         html.H1()
                         ])

performance_layout = html.Div([ html.H1(),
                                html.Div([dcc.Dropdown(id='benchmark', options=[{'label': 'None', 'value': 'None'}, {'label': 'S&P500', 'value': 'S&P500'}, {'label': 'VIX', 'value': 'VIX'}, {'label': 'VVIX', 'value': 'VVIX'}], placeholder='Benchmark')
                                         ]),
                                html.H1(),
                                dcc.Loading(id="loading_graph_base", children=[html.Div(id="output_graph_base")], type="default"),
                                html.H1(),
                                dcc.Loading(id="loading_graph_daily_return", children=[html.Div(id="output_graph_daily_return")], type="default"),
                                html.H1(), 
                                dcc.Loading(id="loading_graph_rolling_perf", children=[html.Div(id="output_graph_rolling_perf")], type="default"),
                                html.H1(), 
                                dcc.Loading(id="loading_graph_dd", children=[html.Div(id="output_graph_dd")], type="default"),
                                html.H1(),
                                dcc.Loading(id="loading_graph_delta", children=[html.Div(id="output_graph_delta")], type="default"),
                                html.H1(),
                                dcc.Loading(id="loading_table_monthly_return", children=[html.Div(id="output_table_monthly_return")], type="default"),
                                html.H1(),
                                dcc.Loading(id="loading_table_stats", children=[html.Div(id="output_table_stats")], type="default")
                             ])

tabs_styles = {'height': '44px'}
tab_style = {'borderBottom': '1px solid #d6d6d6', 'padding': '6px', 'fontWeight': 'bold'}
tab_selected_style = {'borderTop': '1px solid #d6d6d6', 'borderBottom': '1px solid #d6d6d6', 'backgroundColor': '#119DFF', 'color': 'white', 'padding': '6px'}

app.layout = html.Div([ html.Img(src='/assets/favicon.png'),
                        dcc.Tabs(id="tabs", style=tabs_styles, children=[dcc.Tab(label='Portfolio', children=[strategy_layout], style=tab_style, selected_style=tab_selected_style),
                                                                         dcc.Tab(label='Trade', children=[trade_layout], style=tab_style, selected_style=tab_selected_style),
                                                                         dcc.Tab(label='Performance', children=[performance_layout], style=tab_style, selected_style=tab_selected_style)])])

def find_id_table(selected_rows, rows):
    """Return Strategy id of the selected rows
    """
    strategies_id = []
    for ind in selected_rows:
        strategies_id.append(rows[ind]['Strategy nᵒ'])
    strategies_id = list(filter(str.strip, sorted(set(strategies_id), key=strategies_id.index)))
    return strategies_id

def find_id_weight(weights):
    """Return weight id of the selected rows
    """
    weights_id = []
    for w in weights:
        weights_id.append(w['props']['placeholder'][16:])
    return weights_id

def del_weights(weights, del_index):
    """Delete weight cells for the corresponding strategies that has been removed.
    """
    weights = [i for j,i in enumerate(weights) if j not in del_index]
    return weights

def add_weights(weights, add_weight):
    """Add weight cells for the corresponding strategies that has been added.
    """
    for i in add_weight:
        weights.append(dcc.Input(id='w'+str(i), placeholder='Weight Strategy '+str(i)))
    return weights

@app.callback(Output('table_strategies', 'data'), [Input('button_instrument', 'n_clicks')], [State('table_strategies', 'data'), State('table_strategies', 'columns')])
def update_strategies(click, rows, columns):
    """Add a raw in strategy table.
    """
    if (click==0):
        raise PreventUpdate
    else:
        new_instrument = {col['id']: '' for col in columns}
        rows.append(new_instrument)
        return rows

def ids_selected(selected_rows, rows):
    ids = []
    for i in selected_rows:
        strategy_id = rows[i]['Strategy nᵒ']
        instrument_id = rows[i]['Instrument nᵒ']
        ids.append((strategy_id, instrument_id))
    return ids

def update_upload_data(selected_rows, rows, div_upload_data):
    ids = ids_selected(selected_rows, rows)
    if(not div_upload_data):
        return [(ids[0][0], ids[0][1])], None
    else:
        ids_upload = []
        ids_to_add, ids_to_del = [], [] 
        for i in div_upload_data:
            res = i['props']['children'][0]['props']['data'][0]
            ids_upload.append((res['1'][11:], res['2'][11:]))
        for ind,i in enumerate(ids_upload):
            if(i not in ids):
                ids_to_del.append(ind)
        for i in ids:
            if(i not in ids_upload):
                ids_to_add.append(i)
    return ids_to_add, ids_to_del

@app.callback([Output('weights', 'children'), Output('select_before_id', 'message'), Output('select_before_id', 'displayed'), Output('benchmark', 'options'), Output('div_upload_data', 'children'), Output('adjustment', 'dropdown')], [Input('table_strategies', 'selected_rows')], [State('table_strategies', 'data'), State('weights', 'children'), State('benchmark', 'options'), State('div_upload_data', 'children'), State('adjustment', 'dropdown')])
def update_weight(selected_rows, rows, weights, benchmark_option, div_upload_data, adj_dropdown):
    """Add weight strategy for new strategy selected + check validity selection.
    """
    if(selected_rows is None):
        raise PreventUpdate
    select_before_id = [rows[select]['Strategy nᵒ']==str() for select in selected_rows]
    if(any(select_before_id)):
        return weights, 'You must fill the box "Strategy nᵒ" before selecting the strategy. Please fill the box, then deselect and reselect the strategy. A box should appear above the table.', True, benchmark_option, div_upload_data, adj_dropdown
    ids = ids_selected(selected_rows, rows)
    if(len(set(ids))!=len(ids)):
        return weights, 'Tuple (strategy n°, instrument n°) already exits. Must be unique.', True, benchmark_option, div_upload_data, adj_dropdown
    else:
        ids_to_add, ids_to_del = update_upload_data(selected_rows, rows, div_upload_data)
        if(ids_to_del):
            div_upload_data = [i for ind,i in enumerate(div_upload_data) if ind not in ids_to_del]
        if(ids_to_add):
            if(not div_upload_data):
               div_upload_data = []
            [div_upload_data.append(upload_data(i[0], i[1])) for i in ids_to_add]
        strategies_id = find_id_table(selected_rows, rows)
        benchmark_option = benchmark_option[0:3] + [{'label': 'Strategy nᵒ'+i[0], 'value': 'Strategy nᵒ'+i[0]} for i in ids]
        adj_dropdown['Id'] = {'options': [{'label': 'Strategy {} / Instrument {}'.format(i[0], i[1]), 'value': 'Strategy {} / Instrument {}'.format(i[0], i[1])} for i in ids_selected(selected_rows, rows)]}
        if(weights):
            weights_id = find_id_weight(weights)
            del_index = [ind for ind,w in enumerate(weights_id) if w not in strategies_id]
            add_weight = [w for w in strategies_id if w not in weights_id]
            if(del_index):
                weights = del_weights(weights, del_index)
            if(add_weight):
                weights = add_weights(weights, add_weight)
            return weights, 'Ok', False, benchmark_option, div_upload_data, adj_dropdown
        else:
            return [dcc.Input(id='Weight Strategy '+str(i), placeholder='Weight Strategy '+str(i)) for i in strategies_id], 'Ok', False, benchmark_option, div_upload_data, adj_dropdown

@app.callback([Output('current_function_adjustment', 'value'), Output('state_function_adjustment', 'value')], [Input('adjustment', 'data')], [State('current_function_adjustment', 'value')])
def change_in_fct(row, current_fct):
   fct_value = row[0]['Function']
   if(current_fct!=fct_value):
      current_fct = fct_value
      return current_fct, 1
   else:
      return current_fct, 0

@app.callback([Output('div_adjustment', 'children'), Output('div_adjustment', 'style'), Output('table_adjustment', 'data'), Output('upload_file_adjustment', 'message'), Output('upload_file_adjustment', 'displayed')], [Input('button_adjustment', 'n_clicks'), Input('state_function_adjustment', 'value')], [State('table_strategies', 'selected_rows'), State('table_strategies', 'data'), State('button_adjustment', 'n_clicks_timestamp'), State('div_adjustment', 'children'), State('div_adjustment', 'style'), State('adjustment', 'data'), State('table_adjustment', 'data'), State('table_adjustment', 'columns')])
def upload_adjustment(_, change, selected_rows, rows_strategy, n_clicks_timestamp, div_adjustment, style, row, row_info_adj, column_info_adj):
    if(not selected_rows):
        raise PreventUpdate
    elif(n_clicks_timestamp and 1000*time.time()-n_clicks_timestamp<=10):
        ids = ids_selected(selected_rows, rows_strategy)
        row_info_adj = [r for r in row_info_adj if re.findall(r'Strategy ([A-Za-z0-9]+) / Instrument ([A-Za-z0-9]+)', r['Id'])[0] in ids]
        new_adj = {'Id':row[0]['Id'], 'Function':row[0]['Function']}
        if(row[0]['Function']=='Stop Gain'):
            try:
               new_adj['Argument'] = div_adjustment[-2]['props']['value']
            except:
               return div_adjustment, {'columnCount': 2}, row_info_adj, 'Must specify Stop Gain value before applying', True
            new_adj['Argument'] = div_adjustment[-2]['props']['value']
        elif(row[0]['Function']=='Pause'):
            try:
               new_adj['Argument'] = div_adjustment[-3]['props']['filename'] + ' / ' + str(div_adjustment[-2]['props']['value'])
            except:
               return div_adjustment, {'columnCount': 2}, row_info_adj, 'Must specify Pause argument before applying', True
            new_adj['Argument'] = div_adjustment[-3]['props']['filename'] + ' / ' + str(div_adjustment[-2]['props']['value'])
            new_adj['Dates'] = div_adjustment[-3]['props']['contents']
        elif(row[0]['Function'] in ['Keep Contract', 'Stop Contract']):
            try:
               new_adj['Argument'] = div_adjustment[-2]['props']['filename']
            except:
               return div_adjustment, {'columnCount': 2}, row_info_adj, 'Must upload dates file before applying', True
            new_adj['Argument'] = div_adjustment[-2]['props']['filename']
            new_adj['Dates'] = div_adjustment[-2]['props']['contents']
        row_info_adj.append(new_adj)
        return div_adjustment, {'columnCount': 2}, row_info_adj, 'Ok', False
    elif(change==1):
        ids = ids_selected(selected_rows, rows_strategy)
        row_info_adj = [r for r in row_info_adj if re.findall(r'Strategy ([A-Za-z0-9]+) / Instrument ([A-Za-z0-9]+)', r['Id'])[0] in ids]
        if(row[0]['Function']=='Stop Gain'):
            div_adjustment = [div_adjustment[0], dcc.Input(id='stop_gain_val', type='number', placeholder='Stop Gain multiplicator', min=1), div_adjustment[-1]]
            return div_adjustment, {'columnCount': 3}, row_info_adj, 'Ok', False
        div = [dcc.Upload(id='dates', children=html.Div(['Dates (see template file for format):',html.A("""Select File""")]),style={'height': '43px', 'lineHeight': '43px','borderWidth': '1px', 'borderStyle': 'dashed','borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'})]
        if(row[0]['Function']=='Pause'):
            div.append(dcc.Input(id='pause_day_val', type='number', placeholder='Number of days', min=1, step=1))
            div_adjustment = [div_adjustment[0]] + div + [div_adjustment[-1]]
            return div_adjustment, {'columnCount': 4}, row_info_adj, 'Ok', False
        elif(row[0]['Function'] in ['Keep Contract', 'Stop Contract']):
            div_adjustment = [div_adjustment[0]] + div + [div_adjustment[-1]]
            return div_adjustment, {'columnCount': 3}, row_info_adj, 'Ok', False
        else:  
            raise PreventUpdate
    else:
        raise PreventUpdate

@app.callback([Output('confirm', 'message'), Output('confirm', 'displayed')], [Input('button_portfolio', 'n_clicks')], [State('table_strategies', 'data'), State('table_strategies', 'selected_rows'), State('weights', 'children'), State('div_upload_data', 'children')])
def check_valdity_strategy(click, rows, selected_rows, weights, div_upload_entry_weight):
    """Check if all the boxes has been filled before running the strategy.
    """
    if(click is None):
        raise PreventUpdate
    if(selected_rows is None):
        return 'No strategy is selected', True
    for w in weights:
        try:
            w['props']['value']
        except KeyError:
            return 'Please fill the Weights Strategies boxes', True
    for ind in selected_rows:
        if(rows[ind]['Index']=='10Ybond' and rows[ind]['Contract'] in ['Put', 'Call']):
            return 'No option contract exists for 10Y bond.', True
        blank_boxe = [rows[ind][col]==str() for col in columns[2:]]
        if(any(blank_boxe)):
            return 'Please fill the boxes', True
        if(not isinstance(rows[ind]['Maturity'], int)):
            return 'Maturity must be a non negative integer', True
        if(rows[ind]['Maturity']<=0):
            return 'Maturity must be a non negative integer', True
        if(rows[ind]['Delta']>1 or rows[ind]['Delta']<-1):
            return 'Delta must be in [-1,1]', True
    for ind in div_upload_entry_weight:
        try:
            entry_content = ind['props']['children'][1]['props']['contents']
            entry_filename = ind['props']['children'][1]['props']['filename']
        except:
            return 'No Entry date file has been uploaded.', True
        try:
            weight_content = ind['props']['children'][2]['props']['contents']
            weight_filename = ind['props']['children'][2]['props']['filename']
        except:
            return 'No Weight file has been uploaded.', True
    return 'Ok', False

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        return pd.read_excel(io.BytesIO(decoded))

def concate_info(rows, selected_rows, weight, div_upload_entry_weight, row_adjustment):
    strategies = {}
    for ind in selected_rows:
       strategy_id = rows[ind]['Strategy nᵒ']
       instrument_id = rows[ind]['Instrument nᵒ']
       if(strategy_id not in strategies.keys()):
           strategies[strategy_id] = {}
       strategies[strategy_id][instrument_id] = {}
       for col in rows[ind].keys():
          strategies[strategy_id][instrument_id][col] = rows[ind][col]
    for ind in div_upload_entry_weight:
       entry_content = ind['props']['children'][1]['props']['contents']
       entry_filename = ind['props']['children'][1]['props']['filename']
       entry = parse_contents(entry_content, entry_filename).Date.values
       entry = [str_to_datetime(str(date)[0:10]) for date in entry]
       weight_content = ind['props']['children'][2]['props']['contents']
       weight_filename = ind['props']['children'][2]['props']['filename']
       weight_instrument = parse_contents(weight_content, weight_filename)
       weight_instrument = pd.Series([float(w) for w in weight_instrument.Weight.values], index=[str_to_datetime(str(date)[0:10]) for date in weight_instrument.Date])
       strategy_id = ind['props']['children'][0]['props']['data'][0]['1'][11:]
       instrument_id = ind['props']['children'][0]['props']['data'][0]['2'][13:]
       strategies[strategy_id][instrument_id]['Entry'] = entry
       strategies[strategy_id][instrument_id]['Weight'] = weight_instrument
    for ind in row_adjustment:
       Id = re.findall(r'Strategy ([A-Za-z0-9]+) / Instrument ([A-Za-z0-9]+)', ind['Id'])[0]
       strategy_id = Id[0]
       instrument_id = Id[1]
       function = ind['Function']
       strategies[strategy_id][instrument_id]['Adjustment'] = {}
       strategies[strategy_id][instrument_id]['Adjustment'][function] = {}
       if(function=='Stop Gain'):
          strategies[strategy_id][instrument_id]['Adjustment'][function]['Value'] = int(ind['Argument'])
       elif(function=='Pause'):
          filename, time_pause = re.findall(r'([A-Za-z0-9]+) / ([A-Za-z0-9]+)', ind['Argument'])[0]
          strategies[strategy_id][instrument_id]['Adjustment'][function]['Value'] = time_pause
          dates_pause = parse_contents(ind['Dates'], filename).Date.values
          dates_pause = [str_to_datetime(str(date)[0:10]) for date in dates_pause]
          strategies[strategy_id][instrument_id]['Adjustment'][function]['Dates'] = dates_pause
       else:
          dates_pause = parse_contents(ind['Dates'], ind['Argument']).Date.values
          dates_pause = [str_to_datetime(str(date)[0:10]) for date in dates_pause]
          strategies[strategy_id][instrument_id]['Adjustment'][function]['Dates'] = dates_pause
    weight_strategy = {}
    for w in weight:
        strategy_id = w['props']['placeholder'][16:]
        w = int(w['props']['value'])
        weight_strategy[strategy_id] = w
    return strategies, weight_strategy

@app.callback([Output('output_table_trades', 'children'), Output('output_trade_instrument', 'children'), Output('output_graph_base', 'children'), Output('output_graph_daily_return', 'children'), Output('output_graph_rolling_perf', 'children'), Output('output_graph_dd', 'children'), Output('output_graph_delta', 'children'), Output('output_table_monthly_return', 'children'), Output('output_table_stats', 'children')], [Input('confirm', 'message'), Input('benchmark', 'value')], [State('button_portfolio', 'n_clicks_timestamp'), State('table_strategies', 'data'), State('table_strategies', 'selected_rows'), State('weights', 'children'), State('output_table_trades', 'children'), State('output_trade_instrument', 'children'), State('output_graph_base', 'children'), State('output_graph_daily_return', 'children'), State('output_graph_rolling_perf', 'children'), State('output_graph_dd', 'children'), State('output_graph_delta', 'children'), State('output_table_monthly_return', 'children'), State('output_table_stats', 'children'), State('div_upload_data', 'children'), State('table_adjustment', 'data')])
def update_trades(comfirm, selected_benchmark, time_button_portfolio, rows, selected_rows, weights, portfolio_trades, trade_instrument, performance, daily_return, rolling_performance, drawdown, delta, monthly_return, statistics, div_upload_entry_weight, row_adjustment):
    """Run the portfolio.
    """
    if(selected_rows is None or comfirm!='Ok'):
        raise PreventUpdate
    if(1000*time.time()-time_button_portfolio<=5000):
        strategies, weight_strategy = concate_info(rows, selected_rows, weights, div_upload_entry_weight, row_adjustment)
        portfolio = []
        for strategy_id,strategy in strategies.items():
            current_strategy = Strategy(strategy_id)
            for instrument_id,instrument in strategy.items():
                if(instrument['Index']=='S&P'):
                    if(instrument['Contract']=='Call'):
                        tp = 'o'
                        financial_index = spx_call
                    elif(instrument['Contract']=='Put'):
                        tp = 'o'
                        financial_index = spx_put
                    elif(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = spx_future
                elif(instrument['Index']=='Vix'):
                    if(instrument['Contract']=='Call'):
                        tp = 'o'
                        financial_index = vix_call
                    elif(instrument['Contract']=='Put'):
                        tp = 'o'
                        financial_index = vix_put
                    elif(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = vix_future
                elif(instrument['Index']=='10Ybond'):
                    if(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = bond_future
                entry, maturity, position, entry_type, rebalance, delta, weight = instrument['Entry'], int(instrument['Maturity']), int(instrument['Position']), instrument['Entry Type'], instrument['End'], float(instrument['Delta']), instrument['Weight']
                current_strategy.add_instrument(instrument_id, financial_index, entry, maturity, position, tp, rebalance, delta)
                current_strategy.type_investment(instrument_id, entry_type)
                current_strategy.weights(instrument_id, weight)
                if('Adjustment' in instrument.keys()):
                    if('Pause' in instrument['Adjustment']):
                        current_strategy.pause(instrument_id, instrument['Adjustment']['Pause']['Dates'], instrument['Adjustment']['Pause']['Value'])
                    if('Stop Gain' in instrument['Adjustment']):
                      current_strategy.adjust_instrument(instrument_id, 'stop_gain', instrument['Adjustment']['Stop Gain']['Value'])
                    if('Keep Contract' in instrument['Adjustment']):
                      current_strategy.adjust_instrument(instrument_id, 'keep_trade', instrument['Adjustment']['Keep Contract']['Dates'])
                    if('Stop Contract' in instrument['Adjustment']):
                      current_strategy.adjust_instrument(instrument_id, 'stop_date', instrument['Adjustment']['Stop Contract']['Dates'])
            portfolio.append(current_strategy)
        portfolio = Portfolio(*portfolio)
        portfolio.weights(weight_strategy)
        portfolio.fit()
        portfolio.portfolio.daily_return, portfolio.portfolio.quantity = portfolio.portfolio.daily_return.round(6), portfolio.portfolio.quantity.round(6)
        portfolio.portfolio.base = portfolio.portfolio.base.round(3)
        if(not set(portfolio.portfolio.type)=={'f'}):
            portfolio.portfolio.delta = portfolio.portfolio.delta.round(2)
        portfolio.portfolio['date'] = portfolio.portfolio.index
        portfolio_trades =  dash_table.DataTable( id='portfolio_trades',
                                                  data = portfolio.portfolio.to_dict('records'),
                                                  columns=[{'id': i, 'name': i} for i in ['date', 'base', 'daily_return', 'strategy_id', 'instrument_id', 'type', 'position', 'trade_id', 'weight', 'weight_instrument', 'weight_strategy', 'weight', 'quantity', 'maturity', 'underlying']],
                                                  style_cell={'textAlign': 'left'},
                                                  virtualization=True,
                                                  page_action='none', 
                                                  export_format='xlsx',
                                                  filter_action="native",
                                                  sort_action="native",
                                                  sort_mode="multi")
        column_instruments = ['date', 'value', 'maturity', 'underlying', 'position', 'type', 'first_trading_day', 'trade_id', 'P&L_trade', 'instrument_id', 'type_investment', 'weight_instrument', 'strategy_id', 'weight_strategy', 'weight']
        for strategy_id, strategy in portfolio.strategies.items():
            trade_instrument = []
            for instrument_id, instrument in strategy.instruments.items():
              trade_instrument.append(html.H4(children='Strategy n°'+strategy_id+' / '+'Instrument n°'+instrument_id, style={'color': '#0073aa'}))
              trade_instrument.append(dash_table.DataTable( id='strategy'+str(strategy_id)+'_'+'instrument'+str(instrument_id),
                                                            data=instrument.reset_index().to_dict('records'), 
                                                            columns=[{'id': col, 'name': col} for col in column_instruments],
                                                            style_cell={'textAlign': 'left'},
                                                            virtualization=True,
                                                            page_action='none', 
                                                            export_format='xlsx',
                                                            filter_action="native",
                                                            sort_action="native",
                                                            sort_mode="multi"))
        performance =  dcc.Graph(figure={'data': [{'x': portfolio.portfolio.date, 'y': portfolio.portfolio.base, 'type': 'line', 'name':'Portfolio'}], 'layout': {'title': 'Performance'}})
        daily_return = dcc.Graph(figure={'data': [{'x': portfolio.portfolio.date, 'y': portfolio.portfolio.daily_return, 'type': 'line'}], 'layout': {'title': 'Daily Return'}})
        rolling_performance = dcc.Graph(figure={'data': [{'x': portfolio.portfolio.date, 'y': portfolio.extract_daily_return().rolling(252).apply(lambda x: (1+x).prod()-1), 'type': 'line'}], 'layout': {'title': 'Rolling 12 month Performance'}})
        drawdown = dcc.Graph(figure={'data': [{'x': portfolio.portfolio.date, 'y': portfolio.portfolio.base/portfolio.portfolio.base.cummax()-1, 'type': 'line'}], 'layout': {'title': 'Drawdown'}}) 
        delta = dcc.Graph(figure={'data': [{'x': portfolio.portfolio.date, 'y': portfolio.portfolio.delta, 'type': 'line'}], 'layout': {'title': 'Delta'}})
        monthly_return = dash_table.DataTable(id='table_monthly_return',
                                              data=portfolio.monthly_return().reset_index().rename(columns={'index':'Date'}).to_dict('records'),
                                              columns=[{'id': i, 'name': i} for i in portfolio.monthly_return().reset_index().rename(columns={'index':'Date'}).columns],
                                              style_cell={'textAlign': 'left'})
        statistics = dash_table.DataTable( id='table_stats',
                                           data=portfolio.performance_statistics().to_dict('records'),
                                           columns=[{'id': i, 'name': i} for i in ['Max DD', 'DSharpe', 'Dvol', 'MSharpe', 'Mvol', 'Ann. Return']],
                                           style_cell={'textAlign': 'left'})
        return portfolio_trades, trade_instrument, performance, daily_return, rolling_performance, drawdown, delta, monthly_return, statistics
    else:
        data_portfolio = performance['props']['figure']['data'][0]
        if(selected_benchmark=='None'):
            performance = dcc.Graph(figure={'data': [data_portfolio], 'layout': {'title': 'Portfolio'}})
            return portfolio_trades, trade_instrument, performance, daily_return, rolling_performance, drawdown, delta, monthly_return, statistics
        elif(selected_benchmark=='S&P500'):
            benchmark = spx_spot.copy()
        elif(selected_benchmark=='VIX'):
            benchmark = vix_spot.copy()
        else:
            strategies, weight_strategy = concate_info(rows, selected_rows, weights, div_upload_entry_weight, row_adjustment)
            benchmark_id = selected_benchmark[11:]
            benchmark_strategy = Strategy(benchmark_id)
            for instrument_id, instrument in strategies[benchmark_id].items():
                if(instrument['Index']=='S&P'):
                    if(instrument['Contract']=='Call'):
                        tp = 'o'
                        financial_index = spx_call
                    elif(instrument['Contract']=='Put'):
                        tp = 'o'
                        financial_index = spx_put
                    elif(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = spx_future
                elif(instrument['Index']=='Vix'):
                    if(instrument['Contract']=='Call'):
                        tp = 'o'
                        financial_index = vix_call
                    elif(instrument['Contract']=='Put'):
                        tp = 'o'
                        financial_index = vix_put
                    elif(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = vix_future
                elif(instrument['Index']=='10Ybond'):
                    if(instrument['Contract']=='Future'):
                        tp = 'f'
                        financial_index = bond_future
                entry, maturity, position, entry_type, rebalance, delta, weight = instrument['Entry'], int(instrument['Maturity']), int(instrument['Position']), instrument['Entry Type'], instrument['End'], float(instrument['Delta']), instrument['Weight']
                benchmark_strategy.add_instrument(instrument_id, financial_index, entry, maturity, position, tp, rebalance, delta)
                benchmark_strategy.type_investment(instrument_id, entry_type)
                benchmark_strategy.weights(instrument_id, weight)
                if('Adjustment' in instrument.keys()):
                    if('Pause' in instrument['Adjustment']):
                        benchmark_strategy.pause(instrument_id, instrument['Adjustment']['Pause']['Dates'], instrument['Adjustment']['Pause']['Value'])
                    if('Stop Gain' in instrument['Adjustment']):
                        benchmark_strategy.adjust_instrument(instrument_id, 'stop_gain', instrument['Adjustment']['Stop Gain']['Value'])
                    if('Keep Contract' in instrument['Adjustment']):
                        benchmark_strategy.adjust_instrument(instrument_id, 'keep_trade', instrument['Adjustment']['Keep Contract']['Dates'])
                    if('Stop Contract' in instrument['Adjustment']):
                        benchmark_strategy.adjust_instrument(instrument_id, 'stop_date', instrument['Adjustment']['Stop Contract']['Dates'])
            benchmark = Portfolio(benchmark_strategy)
            benchmark.weights(weight_strategy)
            benchmark.fit()
            benchmark.portfolio.daily_return, benchmark.portfolio.quantity = benchmark.portfolio.daily_return.round(6), benchmark.portfolio.quantity.round(6)
            benchmark.portfolio.base = benchmark.portfolio.base.round(3)
            if(not set(benchmark.portfolio.type)=={'f'}):
                benchmark.portfolio.delta = benchmark.portfolio.delta.round(2)
            benchmark.portfolio['date'] = benchmark.portfolio.index
            benchmark.portfolio = benchmark.portfolio.loc[benchmark.portfolio.date>=str_to_datetime(data_portfolio['x'][0])]
            data_benchmark = {'x': benchmark.portfolio.date, 'y': benchmark.portfolio.base, 'type': 'line', 'name':'Benchmark'}
            performance = dcc.Graph(figure={'data': [data_portfolio]+[data_benchmark], 'layout': {'title': 'Performance'}})
            return portfolio_trades, trade_instrument, performance, daily_return, rolling_performance, drawdown, delta, monthly_return, statistics
        benchmark = benchmark.loc[benchmark.index>=str_to_datetime(data_portfolio['x'][0])]
        benchmark.close = benchmark.close/benchmark.close[0]
        data_benchmark = {'x': benchmark.index, 'y': benchmark.close, 'type': 'line', 'name':'Benchmark'}
        performance = dcc.Graph(figure={'data': [data_portfolio]+[data_benchmark], 'layout': {'title': 'Performance'}})
        return portfolio_trades, trade_instrument, performance, daily_return, rolling_performance, drawdown, delta, monthly_return, statistics

def import_data():
    backup = pd.HDFStore('data.h5')
    spx_future = backup['spx_future']
    spx_put = backup['spx_put']
    spx_call = backup['spx_call']
    vix_future = backup['vix_future']
    vix_put = backup['vix_put']
    vix_call = backup['vix_call']
    bond_future = backup['bond_future']
    return spx_future, spx_put, spx_call, vix_future, vix_put, vix_call, bond_future

if __name__ == '__main__':
    spx_future, spx_put, spx_call, vix_future, vix_put, vix_call, bond_future = import_data()
    app.run_server()