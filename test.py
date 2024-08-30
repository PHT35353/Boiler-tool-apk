import pandas as pd
import streamlit as st
from entsoe import EntsoePandasClient
from decimal import Decimal
from scipy.signal import find_peaks
import plotly.graph_objs as go
import numpy as np

# required own api token from entsoe
API_TOKEN = '0464a296-1b5d-4be6-a037-b3414de630f8'
client = EntsoePandasClient(api_key=API_TOKEN)
st.title('Boiler Efficiency and Power Analysis Tool')
# the requested explanation which elaborates how to use the apk
st.markdown("""
### How it Works:

**Purpose:** Compare the efficiency and costs of E-boilers vs. Gas-boilers based on day-ahead and imbalance electricity prices

**Inputs:**
- **Date Range:** Select start and end dates of the time period that you wish to have a data of
- **Country:** Enter the country code of the country that you wish to get the data from
- **Gas Price:** Input the gas price (EUR/kWh) 
- **Desired Power:** Enter your desired power output (kW) that you wish to get out of the boilers

**Outputs:**
- Results are shown in interactive plots with a summary of key findings

**Running the Tool:**
- A running icon appears next to the “Stop” button in the top right while the tool is processing

**Stopping the Tool:**
- Click “Stop” in the top right to stop the tool

**Note:**
- Longer date ranges may increase processing time
- Please ensure your Excel file's time columns are labeled as 'Start time' and 'End time', and the power column is labeled as 'Thermal load (kW)' before uploading

**Settings:**
- Adjust theme or enable wide mode via the settings (three dots at the top right)
""")


# this function gets the day-ahead prices from entsoe
def get_day_ahead_data(start, end, country_code):

    # putting the start and end time to CET
    start = pd.Timestamp(start, tz='Europe/Brussels')
    end = pd.Timestamp(end, tz='Europe/Brussels')

    # getting day-ahead prices
    day_ahead_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    day_ahead_prices = day_ahead_prices.reset_index()
    day_ahead_prices.columns = ['Time', 'Day-Ahead_Price_EUR_per_MWh']

    return day_ahead_prices


# this function gets the imbalance prices from entsoe
def get_imbalance_data(start, end, country_code):
    start = pd.Timestamp(start, tz='Europe/Brussels')
    end = pd.Timestamp(end, tz='Europe/Brussels')

    # getting imbalance prices
    imbalance_prices = client.query_imbalance_prices(country_code, start=start, end=end)
    imbalance_prices = imbalance_prices.reset_index()

    # replacing the name of the time column to time and putting a saftey meassure that gives error if such column does not exist
    if 'index' in imbalance_prices.columns:
        imbalance_prices.rename(columns={'index': 'Time'}, inplace=True)
    else:
        st.error("The time column was not found in the imbalance prices data")
        return pd.DataFrame()

     # this combines the two imbalance columns
    if 'Long' in imbalance_prices.columns and 'Short' in imbalance_prices.columns:
        imbalance_prices['Imbalance_Price_EUR_per_MWh'] = imbalance_prices[['Long', 'Short']].mean(axis=1)
    else:
        st.error("The surplus and shortage data columns are not found in imbalance data")
        return pd.DataFrame()

    imbalance_prices = imbalance_prices[['Time', 'Imbalance_Price_EUR_per_MWh']]

    return imbalance_prices


# this function checks when is either E-boiler or gas-boiler efficient for the day-ahead data
def efficient_boiler_day_ahead(day_ahead_price, gas_price):
    if pd.isna(day_ahead_price):
        return 'Unknown' # if there are no data's it gives unknown
    if Decimal(day_ahead_price) < Decimal(gas_price) / 1000: #Decimal is used for more precise determining
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# this function checks when is either E-boiler or gas-boiler efficient for the imbalance data
def efficient_boiler_imbalance(imbalance_price, gas_price):
    if pd.isna(imbalance_price):
        return 'Unknown'
    if Decimal(imbalance_price) < Decimal(gas_price) / 1000:
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# this function applies the efficient_boiler_day_ahead function to the Day-Ahead_Price_EUR_per_MWh column
def day_ahead_costs(data, gas_price):
    data['Efficient_Boiler_Day_Ahead'] = data['Day-Ahead_Price_EUR_per_MWh'].apply(efficient_boiler_day_ahead, gas_price=gas_price)
    return data

# this function applies the efficient_boiler_imbalance function to the imbalance_Price_EUR_per_MWh column
def imbalance_costs(data, gas_price):
    data['Efficient_Boiler_Imbalance'] = data['Imbalance_Price_EUR_per_MWh'].apply(efficient_boiler_imbalance, gas_price=gas_price)
    return data


# this function adds the clients desired power as an extra column and it shows the power usage of the efficient boiler from the day-ahead market only
def day_ahead_power(data):
    data['E-boiler_Power_Day_Ahead'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Day_Ahead'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else 0, axis=1)
    return data

# this function adds the clients desired power as an extra column and it shows the power usage of the efficient boiler from the imbalance market only
def imbalance_power(data):
    data['E-boiler_Power_Imbalance'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Imbalance'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0, axis=1)
    return data

# this function automatically calculates the timing pattern of the data
def calculate_time_diff_hours(data):
    # calculates the time difference in minutes between the rows
    data['Time_Diff_Minutes'] = data['Time'].diff().dt.total_seconds() / 60.0

    # it defaults the timing pattern to 15 minutes at the beginning of the data where it can not find the two difference between the times
    if data['Time_Diff_Minutes'].isnull().any():
        default_time_diff = 15  
        data['Time_Diff_Minutes'].fillna(default_time_diff, inplace=True)

    # converts to hours
    data['Time_Diff_Hours'] = data['Time_Diff_Minutes'] / 60.0

    return data


# this function calculates the total saving price and precentage of the day-ahead market
def calculate_savings_day_ahead(data, gas_price):
    gas_price_Mwh = gas_price * 1000  # convert gas price from EUR/kWh to EUR/MWh

    # ensures 'Desired Power' is numeric and replace NaN values with 0
    data['Desired Power'] = pd.to_numeric(data['Desired Power'], errors='coerce').fillna(0)

    # calculates gas boiler cost if it was the only option
    data['only_gas_boiler_cost'] = (data['Desired Power'] / 1000) * gas_price_Mwh

    # calculates the gas boiler cost based on efficiency
    data['gas_boiler_cost_in_euro_per_hour'] = data.apply(
        lambda row: (row['Desired Power'] / 1000) * gas_price_Mwh
        if row['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else 0, axis=1
    )
    gas_boiler_cost = data['gas_boiler_cost_in_euro_per_hour'].sum()

    # calculates the E-boiler cost based on efficiency
    data['e_boiler_cost_in_euro_per_hour'] = data.apply(
        lambda row: (row['Desired Power'] / 1000) * row['Day-Ahead_Price_EUR_per_MWh']
        if row['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else 0, axis=1
    )
    e_boiler_cost = data['e_boiler_cost_in_euro_per_hour'].sum()

    # calculates total gas boiler cost if it was the only option
    only_gas_boiler_cost = data['only_gas_boiler_cost'].sum()

    # calculates total savings and percentage savings
    total_mixed_cost = e_boiler_cost + gas_boiler_cost
    total_savings = only_gas_boiler_cost - total_mixed_cost
    percentage_savings = (total_savings / only_gas_boiler_cost * 100) if only_gas_boiler_cost else 0

    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost, only_gas_boiler_cost

# this function calculates the total saving price and precentage of the imbalance market
def calculate_savings_imbalance(data, gas_price):
    gas_price_Mwh = gas_price * 1000  # converts gas price from EUR/kWh to EUR/MWh

    # ensures 'Desired Power' is numeric and replace NaN values with 0
    data['Desired Power'] = pd.to_numeric(data['Desired Power'], errors='coerce').fillna(0)

    # calculates gas boiler cost if it was the only option
    data['only_gas_boiler_cost'] = (data['Desired Power'] / 1000) * (data['Time_Diff_Hours']) * gas_price_Mwh

    # calculates the gas boiler cost based on efficiency
    data['gas_boiler_cost_in_euro_per_hour'] = data.apply(
        lambda row: (row['Desired Power'] / 1000) * (row['Time_Diff_Hours']) * gas_price_Mwh
        if row['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0, axis=1
    )
    gas_boiler_cost = data['gas_boiler_cost_in_euro_per_hour'].sum()

    # calculates the E-boiler cost based on efficiency
    data['e_boiler_cost_in_euro_per_hour'] = data.apply(
        lambda row: (row['Desired Power'] / 1000) * (row['Time_Diff_Hours']) * row['Imbalance_Price_EUR_per_MWh']
        if row['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0, axis=1
    )
    e_boiler_cost = data['e_boiler_cost_in_euro_per_hour'].sum()

    # calculates total gas boiler cost if it was the only option
    only_gas_boiler_cost = data['only_gas_boiler_cost'].sum()

    # adjusts only_gas_boiler_cost by adding the missing hour's cost
    if 'Time_Diff_Hours' in data.columns:
        last_gas_boiler_cost = data['only_gas_boiler_cost'].iloc[-1]
        
        adjusted_gas_boiler_cost = only_gas_boiler_cost + (last_gas_boiler_cost * 4)
    else:
        adjusted_gas_boiler_cost = only_gas_boiler_cost

    # calculates total savings and percentage savings
    total_mixed_cost = e_boiler_cost + gas_boiler_cost
    total_savings = adjusted_gas_boiler_cost - total_mixed_cost
    percentage_savings = (total_savings / adjusted_gas_boiler_cost * 100) if adjusted_gas_boiler_cost else 0

    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost, adjusted_gas_boiler_cost, data



# this function calculates per data which market is more profitable
def calculate_market_profits(day_ahead_data, imbalance_data):
   # makes sure the time column exists
    if 'Time' not in imbalance_data.columns:
        st.error("The time column is missing in imbalance_data.")
        return None, None, None
    # ensures the NaN values to be seen as 0
    imbalance_data['Time'] = pd.to_datetime(imbalance_data['Time'])

    # calculates the profits for both markets 
    day_ahead_data['Profit_Day_Ahead'] = day_ahead_data.apply(
        lambda row: row['gas_boiler_cost_in_euro_per_hour'] - abs(row['e_boiler_cost_in_euro_per_hour']) if row['Day-Ahead_Price_EUR_per_MWh'] < 0 else 0, axis=1
    )
    imbalance_data['Profit_Imbalance'] = imbalance_data.apply(
        lambda row: row['gas_boiler_cost_in_euro_per_hour'] - abs(row['e_boiler_cost_in_euro_per_hour']) if row['Imbalance_Price_EUR_per_MWh'] < 0 else 0, axis=1
    )

    # resamples only the Profit_Imbalance column to hourly intervals for comparison table
    imbalance_profit_resampled = imbalance_data.set_index('Time')['Profit_Imbalance'].resample('H').sum().reset_index()

    # merges day-ahead and resampled imbalance data on 'Time' for profit comparison
    combined_data = pd.merge(day_ahead_data[['Time', 'Profit_Day_Ahead']], imbalance_profit_resampled[['Time', 'Profit_Imbalance']], on='Time', how='outer')

    # replacing NaN values with 0 in both columns
    combined_data['Profit_Day_Ahead'].fillna(0, inplace=True)
    combined_data['Profit_Imbalance'].fillna(0, inplace=True)

    # determines the most profitable market
    combined_data['Most_Profitable_Market'] = combined_data.apply(
        lambda row: (
            'Day-Ahead' if row['Profit_Day_Ahead'] < row['Profit_Imbalance']
            else 'Imbalance' if row['Profit_Imbalance'] < row['Profit_Day_Ahead']
            else 'Gas'
        ), axis=1
    )

    return day_ahead_data, imbalance_data, combined_data

# this is for plotting the price graph
def plot_price(day_ahead_data, imbalance_data, gas_price):
    
    gas_price_kwh = gas_price

    # day-Ahead data processing
    if 'Day-Ahead_Price_EUR_per_MWh' in day_ahead_data.columns:

        # ensures the necessary columns exist before processing
        if 'Efficient_Boiler_Day_Ahead' in day_ahead_data.columns:
            day_ahead_data['E_Boiler_Price_EUR_per_KWh'] = day_ahead_data.apply(
                lambda row: (row['Day-Ahead_Price_EUR_per_MWh'] / 1000) if row['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else 0,
                axis=1
            )
            day_ahead_data['Gas_Boiler_Price_EUR_per_KWh'] = day_ahead_data.apply(
                lambda row: gas_price_kwh if row['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else 0,
                axis=1
            )
        else:
            st.error("The Efficient_Boiler_Day_Ahead column is missing in the day_ahead_data DataFrame.")
            return None, None
    else:
        st.error("Day-Ahead_Price_EUR_per_MWh column is missing in day_ahead_data.")
        return None, None

    # imbalance data processing
    if 'Imbalance_Price_EUR_per_MWh' in imbalance_data.columns:
        # ensures the necessary columns exist before processing
        if 'Efficient_Boiler_Imbalance' in imbalance_data.columns and 'Time_Diff_Hours' in imbalance_data.columns:
            imbalance_data['E_Boiler_Price_EUR_per_KWh'] = imbalance_data.apply(
                lambda row: (row['Imbalance_Price_EUR_per_MWh'] / 1000) * row['Time_Diff_Hours'] if row['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0,
                axis=1
            )
            imbalance_data['Gas_Boiler_Price_EUR_per_KWh'] = imbalance_data.apply(
                lambda row: gas_price_kwh * row['Time_Diff_Hours'] if row['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0,
                axis=1
            )
        else:
            st.error("The Efficient_Boiler_Imbalance or Time_Diff_Hours column is missing in the imbalance_data DataFrame.")
            return None, None
    else:
        st.error("Imbalance_Price_EUR_per_MWh column is missing in imbalance_data.")
        return None, None

    # checks for the existence of the required columns before plotting
    required_columns = ['Time', 'E_Boiler_Price_EUR_per_KWh', 'Gas_Boiler_Price_EUR_per_KWh']
    missing_columns = [col for col in required_columns if col not in day_ahead_data.columns or col not in imbalance_data.columns]

    if missing_columns:
        st.error(f"Missing columns in data: {', '.join(missing_columns)}")
        return None, None

    # plots the day-ahead graph
    day_ahead_fig = go.Figure()

    # converts time to datetime
    if 'Time' in day_ahead_data.columns:
        day_ahead_data['Time'] = pd.to_datetime(day_ahead_data['Time'])
    else:
        st.error("Time column is missing in day_ahead_data.")
        return None, None

    # plots the day-ahead E-boiler and Gas-boiler prices
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['E_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='E-boiler Price (Day-Ahead)', line=dict(color='blue')))
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['Gas_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='Gas-boiler Price (Day-Ahead)', line=dict(color='red', dash='dash')))

    # adds titles and labels
    day_ahead_fig.update_layout(title='Day-Ahead E-boiler vs Gas-boiler Prices (EUR/kWh)',
                                xaxis_title='Time',
                                yaxis_title='Price (EUR/kWh)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    # plots the imbalance graph
    imbalance_fig = go.Figure()

    if 'Time' in imbalance_data.columns:
        imbalance_data['Time'] = pd.to_datetime(imbalance_data['Time'])
    else:
        st.error("Time column is missing in imbalance_data.")
        return None, None

    # plots the imbalance E-boiler and Gas-boiler prices
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['E_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='E-boiler Price (Imbalance)', line=dict(color='blue')))
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['Gas_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='Gas-boiler Price (Imbalance)', line=dict(color='red', dash='dash')))

    # adds titles and labels
    imbalance_fig.update_layout(title='Imbalance E-boiler vs Gas-boiler Prices (EUR/kWh)',
                                xaxis_title='Time',
                                yaxis_title='Price (EUR/kWh)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    return day_ahead_fig, imbalance_fig

# this function plots the total number of times that either the gas-boiler or e-boiler is used in the day-ahead or imbalance market and makes a chart with plotly
def plot_power(day_ahead_data, imbalance_data):
    # ensures the power columns are numeric and replace NaN values with 0
    day_ahead_data['E-boiler_Power_Day_Ahead'] = pd.to_numeric(day_ahead_data['E-boiler_Power_Day_Ahead'], errors='coerce').fillna(0)
    day_ahead_data['Gas-boiler_Power_Day_Ahead'] = pd.to_numeric(day_ahead_data['Gas-boiler_Power_Day_Ahead'], errors='coerce').fillna(0)
    imbalance_data['E-boiler_Power_Imbalance'] = pd.to_numeric(imbalance_data['E-boiler_Power_Imbalance'], errors='coerce').fillna(0)
    imbalance_data['Gas-boiler_Power_Imbalance'] = pd.to_numeric(imbalance_data['Gas-boiler_Power_Imbalance'], errors='coerce').fillna(0)

    # calculates the sum of power for day-ahead data
    sum_e_boiler_day_ahead = day_ahead_data['E-boiler_Power_Day_Ahead'].sum()
    sum_gas_boiler_day_ahead = day_ahead_data['Gas-boiler_Power_Day_Ahead'].sum()

    # creates a horizontal bar chart for day-ahead data using Plotly
    day_ahead_fig = go.Figure()
    day_ahead_fig.add_trace(go.Bar(
        y=['E-boiler (Day-Ahead)', 'Gas-boiler (Day-Ahead)'],
        x=[sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))

    # adds titles and labels
    day_ahead_fig.update_layout(title='Total Power - Day-Ahead',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead) * 1.1]))

    # calculates the sum of power for imbalance data
    sum_e_boiler_imbalance = imbalance_data['E-boiler_Power_Imbalance'].sum()
    sum_gas_boiler_imbalance = imbalance_data['Gas-boiler_Power_Imbalance'].sum()

    # creates a horizontal bar chart for imbalance data using Plotly
    imbalance_fig = go.Figure()
    imbalance_fig.add_trace(go.Bar(
        y=['E-boiler (Imbalance)', 'Gas-boiler (Imbalance)'],
        x=[sum_e_boiler_imbalance, sum_gas_boiler_imbalance],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))

    # adds titles and labels
    imbalance_fig.update_layout(title='Total Power - Imbalance',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_imbalance, sum_gas_boiler_imbalance) * 1.1]))

    return day_ahead_fig, imbalance_fig


# this function connects everything to streamlit
def main():
    # sidebar settings
    st.sidebar.title('Settings')
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2024-01-01'))
    country_code = st.sidebar.text_input('Country code', 'NL')
    gas_price = st.sidebar.number_input('Gas price EUR/kWh', value=0.30 / 9.796)
    desired_power = st.sidebar.number_input('Desired Power (kW)', min_value=0.0, value=100.0, step=1.0)
    uploaded_file = st.sidebar.file_uploader("Upload your desired power data (Excel file)", type=["xlsx", "xls"])

    if st.sidebar.button('Get Data'):
        # gets data
        day_ahead_data = get_day_ahead_data(start_date, end_date, country_code)
        imbalance_data = get_imbalance_data(start_date, end_date, country_code)

        # checks if data is empty and display an error if necessary
        if day_ahead_data.empty:
            st.error("No day-ahead data available")
            return
        if imbalance_data.empty:
            st.error("No imbalance data available")
            return

        # checks if its the default or the uploaded data
        use_uploaded_data = False

        # process uploaded file if available
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_excel(uploaded_file)
                if 'Start time' in uploaded_data.columns and 'thermal load (kW)' in uploaded_data.columns:
                    uploaded_data['Start time'] = pd.to_datetime(uploaded_data['Start time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

                    # converts to the same timezone as the ENTSO-E data
                    uploaded_data['Start time'] = uploaded_data['Start time'].dt.tz_localize('Europe/Amsterdam', ambiguous='NaT', nonexistent='NaT')

                    # renames and selects necessary columns
                    uploaded_data.rename(columns={'thermal load (kW)': 'Desired Power'}, inplace=True)
                    uploaded_data = uploaded_data[['Start time', 'Desired Power']]

                    # ensures day_ahead_data and imbalance_data are in the same timezone
                    day_ahead_data['Time'] = day_ahead_data['Time'].dt.tz_convert('Europe/Amsterdam')
                    imbalance_data['Time'] = imbalance_data['Time'].dt.tz_convert('Europe/Amsterdam')

                    # merges uploaded data with day-ahead and imbalance data
                    day_ahead_data = pd.merge(day_ahead_data, uploaded_data, left_on='Time', right_on='Start time', how='left').drop(columns=['Start time'])
                    imbalance_data = pd.merge(imbalance_data, uploaded_data, left_on='Time', right_on='Start time', how='left').drop(columns=['Start time'])

                    # fills missing values by forward and backward filling
                    day_ahead_data['Desired Power'] = day_ahead_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')
                    imbalance_data['Desired Power'] = imbalance_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')

                    # sets the flag to true indicating we are using uploaded data
                    use_uploaded_data = True
                else:
                    st.error("Uploaded file must contain Start tim' and thermal load (kW) columns")
                    return
            except Exception as e:
                st.error(f"Error reading the uploaded file: {str(e)}")
                return
        else:
            # if no file uploaded, use the desired power input
            day_ahead_data['Desired Power'] = desired_power
            imbalance_data['Desired Power'] = desired_power

        # calculates costs and power usage
        day_ahead_data = day_ahead_costs(day_ahead_data, gas_price)
        imbalance_data = imbalance_costs(imbalance_data, gas_price)

        day_ahead_data = day_ahead_power(day_ahead_data)
        imbalance_data = imbalance_power(imbalance_data)

        # calculates time differences for imbalance data
        imbalance_data = calculate_time_diff_hours(imbalance_data)

        # calculates savings for both day-ahead and imbalance data
        total_savings_day_ahead, percentage_savings_day_ahead, e_boiler_cost_day_ahead, gas_boiler_cost_day_ahead, only_gas_boiler_cost_day_ahead = calculate_savings_day_ahead(day_ahead_data, gas_price)
        total_savings_imbalance, percentage_savings_imbalance, e_boiler_cost_imbalance, gas_boiler_cost_imbalance, only_gas_boiler_cost_imbalance, imbalance_data = calculate_savings_imbalance(imbalance_data, gas_price)

        total_cost_day_ahead = (e_boiler_cost_day_ahead) + gas_boiler_cost_day_ahead
        total_cost_imbalance = (e_boiler_cost_imbalance) + gas_boiler_cost_imbalance

        # calculates the profit and determine the most profitable market
        day_ahead_data, imbalance_data_display, combined_data = calculate_market_profits(day_ahead_data, imbalance_data)

        # calculates the total profit from each market
        total_profit_day_ahead = day_ahead_data['Profit_Day_Ahead'].sum()
        total_profit_imbalance = imbalance_data_display['Profit_Imbalance'].sum()

        # determines the most profitable market
        most_profitable_market = 'Day-Ahead' if total_profit_day_ahead < total_profit_imbalance else 'Imbalance'

        # calculates the profit percentage for each market
        if gas_boiler_cost_day_ahead and total_profit_day_ahead != 0:
            profit_percentage_day_ahead = (abs(e_boiler_cost_day_ahead) / gas_boiler_cost_day_ahead) * 100
        else:
            profit_percentage_day_ahead = 0

        if gas_boiler_cost_imbalance and total_profit_imbalance != 0:
            profit_percentage_imbalance = (abs(e_boiler_cost_imbalance) / gas_boiler_cost_imbalance) * 100
        else:
            profit_percentage_imbalance = 0

        # displays the Day-Ahead results
        st.write('### Day-Ahead Data Results:')
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([10, 10, 10, 10, 10, 10])
            col1.write(f"**Total Savings:**\n{total_savings_day_ahead:,.2f} EUR")
            col2.write(f"**Percentage Savings:**\n{percentage_savings_day_ahead:.2f}%")
            col3.write(f"**Total Cost (both E-boiler and gas-boiler used):**\n{total_cost_day_ahead:,.2f} EUR")
            col4.write(f"**E-boiler Cost:**\n{e_boiler_cost_day_ahead:,.2f} EUR")
            col5.write(f"**Gas-boiler Cost (when the efficient choice):**\n{gas_boiler_cost_day_ahead:,.2f} EUR")
            col6.write(f"**Gas-boiler Cost (when only used):**\n{only_gas_boiler_cost_day_ahead:,.2f} EUR")

        st.write('### Day-Ahead Data Table:')
        st.dataframe(day_ahead_data.drop(columns=['E-boiler_Price_EUR_per_KWh', 'Gas-boiler_Price_EUR per_KWh'], errors='ignore'))

        st.write('### Day-Ahead Market Price Comparison:')
        fig_day_ahead_price, _ = plot_price(day_ahead_data, imbalance_data_display, gas_price)
        st.plotly_chart(fig_day_ahead_price)

        st.write('### Day-Ahead Market Power Usage:')
        fig_day_ahead_power, _ = plot_power(day_ahead_data, imbalance_data_display)
        st.plotly_chart(fig_day_ahead_power)

        # displays the Imbalance results
        st.write('### Imbalance Data Results:')
        with st.container():
            col7, col8, col9, col10, col11, col12 = st.columns([10, 10, 10, 10, 10, 10])
            col7.write(f"**Total Savings:**\n{total_savings_imbalance:,.2f} EUR")
            col8.write(f"**Percentage Savings:**\n{percentage_savings_imbalance:.2f}%")
            col9.write(f"**Total Cost (both E-boiler and gas-boiler used):**\n{total_cost_imbalance:,.2f} EUR")
            col10.write(f"**E-boiler Cost:**\n{e_boiler_cost_imbalance:,.2f} EUR")
            col11.write(f"**Gas-boiler Cost (when the efficient choice):**\n{gas_boiler_cost_imbalance:,.2f} EUR")
            col12.write(f"**Gas-boiler Cost (when only used):**\n{only_gas_boiler_cost_imbalance:,.2f} EUR")

        # drops the eur/kwh columns
        imbalance_data_display = imbalance_data_display.drop(columns=['E_Boiler_Price_EUR_per_KWh', 'Gas_Boiler_Price_EUR_per_KWh'], errors='ignore')
        
        st.write('### Imbalance Data Table:')
        st.dataframe(imbalance_data_display)

        st.write('### Imbalance Market Price Comparison:')
        _, fig_imbalance_price = plot_price(day_ahead_data, imbalance_data_display, gas_price)
        st.plotly_chart(fig_imbalance_price)

        st.write('### Imbalance Market Power Usage:')
        _, fig_imbalance_power = plot_power(day_ahead_data, imbalance_data_display)
        st.plotly_chart(fig_imbalance_power)

        # displays total profits, profit percentages, and the most profitable market
        st.write(f"### Most Profitable Market Overall: {most_profitable_market}")
        st.write(f"Total Profit - Day-Ahead: {total_profit_day_ahead:,.2f} EUR ({profit_percentage_day_ahead:.2f}%)")
        st.write(f"Total Profit - Imbalance: {total_profit_imbalance:,.2f} EUR ({profit_percentage_imbalance:.2f}%)")

if __name__ == '__main__':
    main()
