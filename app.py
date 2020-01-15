########################################## Loading the Packages ########################################################

import subprocess
import sys


def install(package):
    """
    :param package: str, name of the package to install in current environment
    :return: None
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Imports the necessary packages if they are not in the current environment
try:
    import dash
except ImportError:
    install("dash==1.7.0")
finally:
    import dash

try:
    import plotly
except ImportError:
    install("plotly")
finally:
    import plotly

try:
    import numpy as np
except ImportError:
    install("numpy")
finally:
    import numpy as np

try:
    import pandas as pd
except ImportError:
    install("pandas")
finally:
    import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

################################################ Loading the Data ######################################################

arrivals = pd.read_csv("arrivals.csv", header=4)

departures = pd.read_csv("departures.csv", header=4)

crime = pd.read_csv("Crime.csv", header=2)  # Intentional homicides (per 100.000 people)

investment = pd.read_csv("Investments.csv")

income = pd.read_csv("income.csv")

################################################## Data Cleaning #######################################################
# There are some rows that represent aggregates of countries, let us identify and remove them
ccRemove = income[income["Region"].isna() & income["IncomeGroup"].isna()]

ccRemove = ccRemove[["Country Code"]]

arrivals.drop(arrivals[arrivals["Country Code"].isin(ccRemove["Country Code"])].index, inplace=True)
departures.drop(departures[departures["Country Code"].isin(ccRemove["Country Code"])].index, inplace=True)
income.drop(income[income["Country Code"].isin(ccRemove["Country Code"])].index, inplace=True)
crime.drop(crime[crime["Country Code"].isin(ccRemove["Country Code"])].index, inplace=True)
investment.drop(investment[investment["Country ISO3"].isin(ccRemove["Country Code"])].index, inplace=True)

# Removing the rows with NaN values throughout the years:
arrivals = arrivals.dropna(thresh=len([year for year in range(1995, 2018)]) + 1)  # thresh=24
departures = departures.dropna(thresh=len([year for year in range(1995, 2018)]) + 1)

# Now we can delete all columns with NaN values:
arrivals = arrivals.dropna(axis=1, how="all")
departures = departures.dropna(axis=1, how="all")
crime = crime.dropna(axis=1, how='all')
investment = investment.dropna(axis=1, how='all')

# Let us drop this unwanted column:
arrivals = arrivals.drop(columns=["Indicator Name"])
departures = departures.drop(columns=["Indicator Name"])

# Let us merge all our datasets into one:
our_df_list = [arrivals, departures]
df = pd.concat(our_df_list)
del our_df_list
df = df.sort_values("Country Name")

# We are only working on the region and income group, so we ditch the other, unwanted, columns:
income = income[["Country Code", "Region", "IncomeGroup"]]

# Merge the 'main' dataframe with the 'income' dataframe:
df = pd.merge(df, income, on="Country Code", right_index=True)

# Reorganizing the column order:
cols = df.columns.tolist()
cols.insert(3, "Region")
cols.insert(4, "IncomeGroup")
df = df.reindex(columns=cols[:28])
del cols

# Final adjustments:
df = df.drop(columns=["Country Code"])
df = df.dropna(thresh=len([year for year in range(1995, 2018)]) + 1)  # thresh=24

arrivals.drop(columns=["Indicator Code", "Country Code"], inplace=True)
departures.drop(columns=["Indicator Code", "Country Code"], inplace=True)

########################################################################################################################

crime.dropna(thresh=5)

# Analyzing this new dataset we can see that there are 2 Indicators and 3 Subindicators, so we just select the capital
# invested in tourism as an Indicator, then we seperate the 3 subindicators in 3 different dataframes:
investment = investment.loc[(investment["Indicator"] == "Capital investment in Travel and Tourism") &
                            (investment["Subindicator Type"] == "US$ in bn (Real prices)")]  # We are only concerned about the volumes themselves

investment.rename(columns={"Country ISO3": "Country Code"}, inplace=True)

investment = pd.merge(investment, income, on="Country Code")
crime = pd.merge(crime, income, on="Country Code")

######################################### Interactive Components #######################################################

country_options = [dict(label=country, value=country) for country in df["Country Name"].unique()]

data_options = [{'label': 'Arrivals', 'value': 'Arrivals'}, {'label': 'Departures', 'value': 'Departures'},
                {'label': 'Crime Rate', 'value': 'Crime Rate'},
                {'label': 'Capital Investment', 'value': 'Capital Investment'}]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Div([
        html.H1('Understanding Trends and New Opportunities in the Tourism Sector'),
        html.H3('via Data Visualization')
    ], id='Title'),

    html.Br(),

    html.Div([

        html.Div([
            html.Label('Select Data'),
            dcc.Dropdown(
                id='data_option',
                options=data_options,
                value='Arrivals'
            ),

            html.Br(),

            html.Label('Select Countries'),
            dcc.Dropdown(
                id='country_option',
                options=country_options,
                value=['China', 'United States', 'United Kingdom', 'France', 'Mexico', 'Portugal'],
                multi=True
            )
        ], className='column1 pretty'),

        html.Div([

            html.Div([

                html.Div([dcc.Graph(id='bar_graph_income')], className='column4 pretty'),

                html.Div([dcc.Graph(id='bar_graph_region')], className='column3 pretty'),

            ], className='2 plots row'),

        ], className='column2')

    ], className='row'),

    html.Br(),

    html.Label('Filter by Year Range'),
    dcc.RangeSlider(
        id='year_range_slider',
        min=min([int(column) for column in df.columns
                 if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                 ]),
        max=max([int(column) for column in df.columns
                 if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                 ]),
        marks={str(i): '{}'.format(str(i)) for i in [year for year in range(1995, 2018)]},
        value=[min([int(column) for column in df.columns
                    if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                    ]),
               max([int(column) for column in df.columns
                    if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                    ])
               ],
        step=1,
    ),

    html.Div([dcc.Graph(id='graph')], className='pretty'),

    html.Br(),

    html.Div([

        html.Div([dcc.Graph(id='choropleth')], className='column3 pretty'),

        html.Div([dcc.Graph(id='sunburst')], className='column4 pretty')

    ], className='row'),

    html.Br(),

    html.Label('Select Year'),
    dcc.Slider(
        id='year_slider',
        min=min([int(column) for column in df.columns
                 if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                 ]),
        max=max([int(column) for column in df.columns
                 if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                 ]),
        marks={str(i): '{}'.format(str(i)) for i in [year for year in range(1995, 2018)]},
        value=max([int(column) for column in df.columns
                   if column not in ['Country Name', 'Indicator Code', 'Region', 'IncomeGroup']
                   ]),
        step=1,
    ),

])

############################################### Callbacks ##############################################################

@app.callback(
    [
        Output('graph', 'figure'),
        Output('bar_graph_income', 'figure'),
        Output('bar_graph_region', 'figure')
    ],
    [
        Input('year_range_slider', 'value'),
        Input('country_option', 'value'),
        Input('data_option', 'value')
    ]
)
def bar_plots(year_range, countries, data):
    """
    """
    data_bar = []
    if data == 'Arrivals':
        temp_df = arrivals
        temp_text = 'Number of Arrivals'
    elif data == 'Departures':  # data == 'Departures'
        temp_df = departures
        temp_text = 'Number of Departures'
    elif data == 'Crime Rate':
        temp_df = crime
        temp_text = 'Intentional Homicides (per 100.000 people)'
    else:  # data == 'Capital Investment'
        temp_df = investment
        temp_text = 'Capital Investment in Travel and Tourism in US$ (in bn)'

    for country in countries:
        df_bar = temp_df.loc[temp_df['Country Name'] == country]

        x_bar = [year for year in range(year_range[0], year_range[-1] + 1)]
        try:
            y_bar = df_bar[[str(year) for year in x_bar]].values[0]
        except IndexError:
            y_bar = []

        data_bar.append(dict(type='scatter',
                             x=x_bar,
                             y=y_bar,
                             line_shape='spline',
                             name=country))

    layout_bar = dict(title=dict(text=temp_text + ' from ' + str(year_range[0]) + ' to ' + str(year_range[-1])),
                      paper_bgcolor='#f9f9f9'
                      )

    if data == 'Arrivals':
        temp_df2 = df.loc[df['Indicator Code'] == 'ST.INT.ARVL', ['Country Name', 'Region', 'IncomeGroup']]
    elif data == 'Departures':
        temp_df2 = df.loc[df['Indicator Code'] == 'ST.INT.DPRT', ['Country Name', 'Region', 'IncomeGroup']]
    elif data == 'Crime Rate':
        temp_df2 = crime
    else:  # data == 'Capital Investment'
        temp_df2 = investment

    country_per_income_group = temp_df2.drop_duplicates(subset='Country Name').groupby(by=['IncomeGroup'])['Country Name'].nunique()
    country_per_region = temp_df2.drop_duplicates(subset='Country Name').groupby(by=['Region'])['Country Name'].nunique()

    income_data = dict(type='bar',
                       x=country_per_income_group.sort_values(axis=0).index,  # Order by income group for better visualization
                       y=country_per_income_group,
                       marker=dict(color='orangered')
                       )

    income_layout = dict(title=dict(text='Number of Countries per Income Group (2017)'))

    region_data = dict(type='bar',
                       x=country_per_region.index,
                       y=country_per_region,
                       marker=dict(color='deepskyblue')
                       )

    region_layout = dict(title=dict(text='Number of Countries per Region (2017)'))

    return [go.Figure(data=data_bar, layout=layout_bar),
            go.Figure(data=income_data, layout=income_layout),
            go.Figure(data=region_data, layout=region_layout)]


@app.callback(
    [
        Output('choropleth', 'figure'),
        Output('sunburst', 'figure')
    ],
    [
        Input('year_slider', 'value'),
        Input('data_option', 'value')
    ]
)
def map_sunburst(year, data):
    """
    """
    if data == 'Arrivals':
        temp_df = arrivals
        temp_text = 'Number of Arrivals: '
        temp_text2 = 'Number of Arrivals '
    elif data == 'Departures':
        temp_df = departures
        temp_text = 'Number of Departures: '
        temp_text2 = 'Number of Departures '
    elif data == 'Crime Rate':
        temp_df = crime
        temp_text = 'Number of Intentional Homicides (per 100.000 people): '
        temp_text2 = 'Number of Intentional Homicides (per 100.000 people) '
    else:  # data == 'Capital Investment'
        temp_df = investment
        temp_text = 'Capital Investment in Travel and Tourism in US$ (in bn): '
        temp_text2 = 'Capital Investment in Travel and Tourism '

    temp_df = temp_df[['Country Name', str(year)]]

    # Choropleth

    data_choropleth = dict(type='choropleth',
                           locations=temp_df['Country Name'],
                           locationmode='country names',
                           z=temp_df[str(year)],
                           text=temp_df['Country Name'],
                           colorscale='YlGnBu',
                           colorbar=dict(title='Number of ' + data),
                           hovertemplate='%{text} <br>' + temp_text + '%{z}',
                           name=''
                           )

    layout_choropleth = dict(geo=dict(scope='world',
                                      projection=dict(type='equirectangular'),
                                      landcolor='black',
                                      lakecolor='white',
                                      showocean=True,
                                      oceancolor='azure',
                                      bgcolor='#f9f9f9'
                                      ),
                             title=dict(text='World ' + temp_text2 + 'Choropleth Map on the Year ' + str(year),
                                        x=.475),
                             paper_bgcolor='#f9f9f9',
                             )

    # Sunburst
    if data == 'Arrivals':
        temp_df2 = df.loc[df['Indicator Code'] == 'ST.INT.ARVL', ['Country Name', 'Region', str(year)]]
        temp_df2.dropna()
    elif data == 'Departures':
        temp_df2 = df.loc[df['Indicator Code'] == 'ST.INT.DPRT', ['Country Name', 'Region', str(year)]]
        temp_df2.dropna()
    elif data == 'Crime Rate':
        temp_df2 = crime
        temp_df2.dropna()
    else:  # data == 'Capital Investment'
        temp_df2 = investment
        temp_df2.dropna()

    df_regions = temp_df2.groupby('Region').sum()

    # Converts the dataframe's data into lists, to be able to plot the sunburst:
    regions_list = df_regions.index.tolist()
    regions_values = df_regions[str(year)].tolist()
    country_list = temp_df2['Country Name'].tolist()
    country_value = temp_df2[str(year)].tolist()
    world_value = [df_regions[str(year)].sum()]
    country_region_mapping = temp_df2['Region'].tolist()

    data_sunburst = go.Sunburst(
        labels=['World'] + regions_list + country_list,
        parents=[''] + ['World'] * len(regions_list) + country_region_mapping,
        values=world_value + regions_values + country_value,
        branchvalues='total'
    )

    layout_sunburst = dict(title=dict(text='World Number of ' + data + ' Sunburst on the Year ' + str(year),
                                      x=.475),
                           paper_bgcolor='#f9f9f9',
                           margin=dict(t=0, l=0, r=0, b=0)
                           )

    return go.Figure(data=data_choropleth, layout=layout_choropleth), \
           go.Figure(data=data_sunburst, layout=layout_sunburst)


if __name__ == "__main__":
    app.run_server(debug=True, port=3004)
