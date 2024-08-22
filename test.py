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
st.markdown("""How it Works:

	•Purpose: Compare the efficiency and costs of E-boilers vs. Gas-boilers based on day-ahead and imbalance electricity prices.
	•Inputs:
	  •Date Range: Select start and end dates.
	  •Country: Enter the country code.
	  •Gas Price: Input the gas price (EUR/kWh).
	  •Desired Power: Enter your desired power output (kWh).
	•Outputs:
	  •Results are shown in interactive plots with a summary of key findings.
	  •Running the Tool:
	  •A running icon appears next to the “Stop” button in the top right while the tool is processing.
	•Stopping the Tool:
	  •Click “Stop” in the top right to stop the tool.
	  •Note: Longer date ranges may increase processing time.
	  •Settings: Adjust theme or enable wide mode via the settings (three dots at the top right).""")

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


def calculate_time_diff_hours(data):
    # Calculate the time difference in minutes between consecutive rows
    data['Time_Diff_Minutes'] = data['Time'].diff().dt.total_seconds() / 60.0
    
    # Set the first row's time difference to a default value (e.g., 15 minutes)
    if data['Time_Diff_Minutes'].isnull().any():
        default_time_diff = 15  # Default to 15 minutes if not available
        data['Time_Diff_Minutes'].fillna(default_time_diff, inplace=True)
    
    # Convert to hours
    data['Time_Diff_Hours'] = data['Time_Diff_Minutes'] / 60.0
    
    return data


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
    # Ensure the 'Day-Ahead_Price_EUR_per_MWh' column exists
    if 'Day-Ahead_Price_EUR_per_MWh' not in data.columns:
        st.error("Day-Ahead price data is missing. Cannot determine efficient boiler.")
        return data

    # Determine the efficient boiler based on day-ahead prices
    def efficient_boiler_day_ahead(day_ahead_price):
        if pd.isna(day_ahead_price):
            return 'Unknown'
        if Decimal(day_ahead_price) < Decimal(gas_price) / 1000:
            return 'E-boiler'
        else:
            return 'Gas-boiler'

    data['Efficient_Boiler_Day_Ahead'] = data['Day-Ahead_Price_EUR_per_MWh'].apply(efficient_boiler_day_ahead)
    
    return data


# this function applies the efficient_boiler_imbalance function to the imbalance_Price_EUR_per_MWh column
def imbalance_costs(data, gas_price):
    # Ensure the 'Imbalance_Price_EUR_per_MWh' column exists
    if 'Imbalance_Price_EUR_per_MWh' not in data.columns:
        st.error("Imbalance price data is missing. Cannot determine efficient boiler.")
        return data

    # Determine the efficient boiler based on imbalance prices
    def efficient_boiler_imbalance(imbalance_price):
        if pd.isna(imbalance_price):
            return 'Unknown'
        if Decimal(imbalance_price) < Decimal(gas_price) / 1000:
            return 'E-boiler'
        else:
            return 'Gas-boiler'

    data['Efficient_Boiler_Imbalance'] = data['Imbalance_Price_EUR_per_MWh'].apply(efficient_boiler_imbalance)
    
    return data



# this function adds the clients desired power as an extra column and it shows the power usage of the efficient boiler from the day-ahead market only
def day_ahead_power(data):
    # Ensure the 'Efficient_Boiler_Day_Ahead' and 'Desired Power' columns exist
    if 'Efficient_Boiler_Day_Ahead' not in data.columns:
        st.error("Efficient boiler data for day-ahead market is missing.")
        return data
    if 'Desired Power' not in data.columns:
        st.error("Desired power data is missing.")
        return data

    # Calculate the power usage based on which boiler is efficient
    data['E-boiler_Power_Day_Ahead'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Day_Ahead'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else 0, axis=1)
    
    return data


# this function adds the clients desired power as an extra column and it shows the power usage of the efficient boiler from the imbalance market only
def imbalance_power(data):
    # Ensure the 'Efficient_Boiler_Imbalance' and 'Desired Power' columns exist
    if 'Efficient_Boiler_Imbalance' not in data.columns:
        st.error("Efficient boiler data for imbalance market is missing.")
        return data
    if 'Desired Power' not in data.columns:
        st.error("Desired power data is missing.")
        return data

    # Calculate the power usage based on which boiler is efficient
    data['E-boiler_Power_Imbalance'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Imbalance'] = data.apply(lambda x: x['Desired Power'] if x['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0, axis=1)
    
    return data


# this function calculates the total saving price and precentage of the day-ahead market
def calculate_savings_day_ahead(data, gas_price, desired_power):
    
    gas_price_Mwh = (gas_price) * (1000) # converting the gas-price from EUR/kwh to EUR/Mwh
    desired_power_Mwh = (desired_power) / (1000)  # converting the desired power from kw to Mw
    
    # this calculates the gas-price in euros and adds these prices to a new column
    data['Gas_Boiler_Cost_in_Euro'] = data.apply(lambda row: desired_power_Mwh * gas_price_Mwh 
                                         if row['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else (0), axis=1)
    
    # this then adds the former mentioned gas_price in Euro column up to get the total gas price
    gas_boiler_cost = data['Gas_Boiler_Cost_in_Euro'].sum()
    
    # this calculates the electricity price in euros and adds these prices to a new column
    data['E_Boiler_Cost_in_Euro'] = data.apply(lambda row: desired_power_Mwh * (row['Day-Ahead_Price_EUR_per_MWh']) 
                                       if row['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else (0), axis=1)
    
    #this then adds the former mentioned electricity price in Euro column up to get the total electricity
    e_boiler_cost = data['E_Boiler_Cost_in_Euro'].sum()
    
    # these two functions calculate the savings and the saving precentage
    total_savings = (abs(e_boiler_cost))
    percentage_savings = ((total_savings) / (gas_boiler_cost) * (100)) if gas_boiler_cost else (0)
    
   
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost

def calculate_savings_imbalance(data, gas_price, desired_power):
    # Calculate the time difference in minutes between consecutive rows
    data['Time_Diff_Minutes'] = data['Time'].diff().dt.total_seconds() / 60.0
    
    # Set the first row's time difference to a default value (e.g., 15 minutes or the mean of other time differences)
    if data['Time_Diff_Minutes'].isnull().any():
        default_time_diff = data['Time_Diff_Minutes'].mean()  # You can change this to 15 if 15 minutes is a common interval
        data['Time_Diff_Minutes'].fillna(default_time_diff, inplace=True)
    
    # Convert gas price from EUR/kWh to EUR/MWh
    gas_price_Mwh = gas_price * 1000  # EUR/MWh
    
    # Convert desired power from kW to MW
    desired_power_Mwh = desired_power / 1000.0  # MW
    
    # Calculate the cost dynamically based on time difference
    data['Gas_Boiler_Cost_Imbalance_in_Euro'] = data.apply(lambda row: (desired_power_Mwh * (row['Time_Diff_Minutes'] / 60)) * gas_price_Mwh 
                                                   if row['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0, axis=1)
    gas_boiler_cost = data['Gas_Boiler_Cost_Imbalance_in_Euro'].sum()
    
    # Calculate the e-boiler cost
    data['E_Boiler_Cost_Imbalance_in_Euro'] = data.apply(lambda row: (desired_power_Mwh * (row['Time_Diff_Minutes'] / 60)) * row['Imbalance_Price_EUR_per_MWh'] 
                                                 if row['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0, axis=1)
    e_boiler_cost = data['E_Boiler_Cost_Imbalance_in_Euro'].sum()
    
    # Calculate total savings and percentage savings
    total_savings = abs(e_boiler_cost)
    percentage_savings = (total_savings / gas_boiler_cost * 100) if gas_boiler_cost else 0
    
    # Return the calculated values and the modified DataFrame
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost, data


def calculate_cost_columns(day_ahead_data, imbalance_data, gas_price):
    # Calculate the cost for the E-boiler and Gas-boiler for day-ahead and imbalance markets

    if 'Day-Ahead_Price_EUR_per_MWh' in day_ahead_data.columns:
        if 'E_Boiler_Cost_in_Euro' not in day_ahead_data.columns:
            day_ahead_data['E_Boiler_Cost_in_Euro'] = day_ahead_data['Day-Ahead_Price_EUR_per_MWh'] * (day_ahead_data['Desired Power'] / 1000)
    else:
        st.error("Day-Ahead price data is missing. Cannot calculate E-Boiler costs.")

    if 'Imbalance_Price_EUR_per_MWh' in imbalance_data.columns:
        if 'E_Boiler_Cost_Imbalance_in_Euro' not in imbalance_data.columns:
            imbalance_data['E_Boiler_Cost_Imbalance_in_Euro'] = imbalance_data['Imbalance_Price_EUR_per_MWh'] * (imbalance_data['Desired Power'] / 1000)
    else:
        st.error("Imbalance price data is missing. Cannot calculate E-Boiler costs.")
    
    return day_ahead_data, imbalance_data


def determine_profitability(day_ahead_data, imbalance_data):
    # Check if cost columns exist before proceeding
    if 'E_Boiler_Cost_in_Euro' not in day_ahead_data.columns or 'E_Boiler_Cost_Imbalance_in_Euro' not in imbalance_data.columns:
        st.error("Required cost columns are missing in the dataframes.")
        return day_ahead_data, imbalance_data

    # Define the profitability based on the E-boiler prices
    def calculate_most_profitable(row):
        if row['E_Boiler_Cost_in_Euro'] == 0 and row['E_Boiler_Cost_Imbalance_in_Euro'] == 0:
            return 'No Profit'
        elif row['E_Boiler_Cost_in_Euro'] > row['E_Boiler_Cost_Imbalance_in_Euro']:
            return 'Imbalance'
        elif row['E_Boiler_Cost_in_Euro'] < row['E_Boiler_Cost_Imbalance_in_Euro']:
            return 'Day-Ahead'
        else:
            return 'Equal'

    # Merge day-ahead and imbalance data on 'Time' to compare
    combined_data = pd.merge(day_ahead_data[['Time', 'E_Boiler_Cost_in_Euro']],
                             imbalance_data[['Time', 'E_Boiler_Cost_Imbalance_in_Euro']],
                             on='Time',
                             how='inner')

    # Determine the most profitable market
    combined_data['Most_Profitable_Market'] = combined_data.apply(calculate_most_profitable, axis=1)

    # Merge this result back into the original day-ahead and imbalance dataframes
    day_ahead_data = day_ahead_data.merge(combined_data[['Time', 'Most_Profitable_Market']], on='Time', how='left')
    imbalance_data = imbalance_data.merge(combined_data[['Time', 'Most_Profitable_Market']], on='Time', how='left')

    return day_ahead_data, imbalance_data

# this is for plotting the price graph
def plot_price(day_ahead_data, imbalance_data, gas_price):
    # Convert gas price to EUR/kWh
    gas_price_kwh = gas_price
    
    # Day-Ahead data processing
    if 'Day-Ahead_Price_EUR_per_MWh' in day_ahead_data.columns:
        
        # Ensure the necessary columns exist before processing
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
            st.error("The 'Efficient_Boiler_Day_Ahead' column is missing in the day_ahead_data DataFrame.")
            return None, None
    else:
        st.error("Day-Ahead_Price_EUR_per_MWh column is missing in day_ahead_data.")
        return None, None

    # Imbalance data processing
    if 'Imbalance_Price_EUR_per_MWh' in imbalance_data.columns:        
        # Ensure the necessary columns exist before processing
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
            st.error("The 'Efficient_Boiler_Imbalance' or 'Time_Diff_Hours' column is missing in the imbalance_data DataFrame.")
            return None, None
    else:
        st.error("Imbalance_Price_EUR_per_MWh column is missing in imbalance_data.")
        return None, None

    # Check for the existence of the required columns before plotting
    required_columns = ['Time', 'E_Boiler_Price_EUR_per_KWh', 'Gas_Boiler_Price_EUR_per_KWh']
    missing_columns = [col for col in required_columns if col not in day_ahead_data.columns or col not in imbalance_data.columns]

    if missing_columns:
        st.error(f"Missing columns in data: {', '.join(missing_columns)}")
        return None, None

    # Plot the day-ahead graph
    day_ahead_fig = go.Figure()

    # Convert time to datetime
    if 'Time' in day_ahead_data.columns:
        day_ahead_data['Time'] = pd.to_datetime(day_ahead_data['Time'])
    else:
        st.error("Time column is missing in day_ahead_data.")
        return None, None

    # Plot the day-ahead E-boiler and Gas-boiler prices
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['E_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='E-boiler Price (Day-Ahead)', line=dict(color='blue')))
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['Gas_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='Gas-boiler Price (Day-Ahead)', line=dict(color='red', dash='dash')))

    # Add titles and labels
    day_ahead_fig.update_layout(title='Day-Ahead E-boiler vs Gas-boiler Prices (EUR/kWh)',
                                xaxis_title='Time',
                                yaxis_title='Price (EUR/kWh)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    # Plot the imbalance graph
    imbalance_fig = go.Figure()

    if 'Time' in imbalance_data.columns:
        imbalance_data['Time'] = pd.to_datetime(imbalance_data['Time'])
    else:
        st.error("Time column is missing in imbalance_data.")
        return None, None

    # Plot the imbalance E-boiler and Gas-boiler prices
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['E_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='E-boiler Price (Imbalance)', line=dict(color='blue')))
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['Gas_Boiler_Price_EUR_per_KWh'],
                                       mode='lines', name='Gas-boiler Price (Imbalance)', line=dict(color='red', dash='dash')))

    # Add titles and labels
    imbalance_fig.update_layout(title='Imbalance E-boiler vs Gas-boiler Prices (EUR/kWh)',
                                xaxis_title='Time',
                                yaxis_title='Price (EUR/kWh)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    return day_ahead_fig, imbalance_fig







# this function plots the total number of times that either the gas-boiler or e-boiler is used in the day-ahead or imbalance market and makes a chart with plotly
def plot_power(day_ahead_data, imbalance_data):

    # calculates the sum of power for day-ahead data
    sum_e_boiler_day_ahead = day_ahead_data['E-boiler_Power_Day_Ahead'].sum()
    sum_gas_boiler_day_ahead = day_ahead_data['Gas-boiler_Power_Day_Ahead'].sum()

    # making a horizontal bar chart for day-ahead data using Plotly
    day_ahead_fig = go.Figure()
    day_ahead_fig.add_trace(go.Bar(
        y=['E-boiler (Day-Ahead)', 'Gas-boiler (Day-Ahead)'],
        x=[sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))
    
    # adds the titels and names
    day_ahead_fig.update_layout(title='Total Power - Day-Ahead',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead) * 1.1]))

    # calculates the sum of power for imbalance data
    sum_e_boiler_imbalance = imbalance_data['E-boiler_Power_Imbalance'].sum()
    sum_gas_boiler_imbalance = imbalance_data['Gas-boiler_Power_Imbalance'].sum()

    # making a horizontal bar chart for imbalance data using Plotly
    imbalance_fig = go.Figure()
    imbalance_fig.add_trace(go.Bar(
        y=['E-boiler (Imbalance)', 'Gas-boiler (Imbalance)'],
        x=[sum_e_boiler_imbalance, sum_gas_boiler_imbalance],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))
    
    # adds the titels and names
    imbalance_fig.update_layout(title='Total Power - Imbalance',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_imbalance, sum_gas_boiler_imbalance) * 1.1]))

    return day_ahead_fig, imbalance_fig


def main():
    st.sidebar.title('Settings')
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2024-01-01'))
    country_code = st.sidebar.text_input('Country code', 'NL')
    gas_price = st.sidebar.number_input('Gas price EUR/kWh', value=0.30 / 9.796)
    desired_power = st.sidebar.number_input('Desired Power (kWh)', min_value=0.0, value=100.0, step=1.0)
    uploaded_file = st.sidebar.file_uploader("Upload your desired power data (Excel file)", type=["xlsx", "xls"])

    if st.sidebar.button('Get Data'):
        # Fetch data
        day_ahead_data = get_day_ahead_data(start_date, end_date, country_code)
        imbalance_data = get_imbalance_data(start_date, end_date, country_code)

        if day_ahead_data.empty or imbalance_data.empty:
            st.error("No data available for the selected period or country.")
            return

        # Process uploaded file if available
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_excel(uploaded_file)
                if 'Time' in uploaded_data.columns and 'Desired Power' in uploaded_data.columns:
                    uploaded_data['Time'] = pd.to_datetime(uploaded_data['Time'])
                    day_ahead_data = pd.merge(day_ahead_data, uploaded_data[['Time', 'Desired Power']], on='Time', how='left')
                    imbalance_data = pd.merge(imbalance_data, uploaded_data[['Time', 'Desired Power']], on='Time', how='left')
                    day_ahead_data['Desired Power'] = day_ahead_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')
                    imbalance_data['Desired Power'] = imbalance_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')
                else:
                    st.error("Uploaded file must contain 'Time' and 'Desired Power' columns.")
                    return
            except Exception as e:
                st.error(f"Error reading the uploaded file: {str(e)}")
                return
        else:
            # If no file uploaded, use the desired power input
            day_ahead_data['Desired Power'] = desired_power
            imbalance_data['Desired Power'] = desired_power

        # Calculate costs and power usage
        day_ahead_data, imbalance_data = calculate_cost_columns(day_ahead_data, imbalance_data, gas_price)
        day_ahead_data = day_ahead_power(day_ahead_data)
        imbalance_data = imbalance_power(imbalance_data)

        # Calculate time differences for imbalance data
        imbalance_data = calculate_time_diff_hours(imbalance_data)

        # Determine which market is more profitable per timestamp
        day_ahead_data, imbalance_data = determine_profitability(day_ahead_data, imbalance_data)

        # Calculate the total profit from each market
        total_profit_day_ahead = day_ahead_data['E_Boiler_Cost_in_Euro'].sum()
        total_profit_imbalance = imbalance_data['E_Boiler_Cost_Imbalance_in_Euro'].sum()
        most_profitable_market = 'Day-Ahead' if total_profit_day_ahead < total_profit_imbalance else 'Imbalance'

        # Display the original results for day-ahead data
        st.write('### Day-Ahead Data Results:')
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([10, 10, 10, 10, 10])
            col1.write(f"**Total Savings:**\n{total_profit_day_ahead:,.2f} EUR")
            col2.write(f"**Percentage Savings:**\n{(total_profit_day_ahead / (total_profit_day_ahead + total_profit_imbalance) * 100):.2f}%")
            col3.write(f"**Total Cost:**\n{total_profit_day_ahead:,.2f} EUR")
            col4.write(f"**E-boiler Cost:**\n{total_profit_day_ahead:,.2f} EUR")
            col5.write(f"**Gas-boiler Cost:**\n{total_profit_day_ahead:,.2f} EUR")

        st.write('### Day-Ahead Data Table:')
        st.dataframe(day_ahead_data)

        # Display the original results for imbalance data
        st.write('### Imbalance Data Results:')
        with st.container():
            col6, col7, col8, col9, col10 = st.columns([10, 10, 10, 10, 10])
            col6.write(f"**Total Savings:**\n{total_profit_imbalance:,.2f} EUR")
            col7.write(f"**Percentage Savings:**\n{(total_profit_imbalance / (total_profit_day_ahead + total_profit_imbalance) * 100):.2f}%")
            col8.write(f"**Total Cost:**\n{total_profit_imbalance:,.2f} EUR")
            col9.write(f"**E-boiler Cost:**\n{total_profit_imbalance:,.2f} EUR")
            col10.write(f"**Gas-boiler Cost:**\n{total_profit_imbalance:,.2f} EUR")

        st.write('### Imbalance Data Table:')
        st.dataframe(imbalance_data)

        # Display total profits and most profitable market
        st.write(f"### Most Profitable Market Overall: {most_profitable_market}")
        st.write(f"Total Profit - Day-Ahead: {total_profit_day_ahead:,.2f} EUR")
        st.write(f"Total Profit - Imbalance: {total_profit_imbalance:,.2f} EUR")

        # Plot the price graphs
        fig_day_ahead_price, fig_imbalance_price = plot_price(day_ahead_data, imbalance_data, gas_price)
        if fig_day_ahead_price is not None and fig_imbalance_price is not None:
            st.write('### Price Comparison:')
            st.plotly_chart(fig_day_ahead_price)
            st.plotly_chart(fig_imbalance_price)
        else:
            st.error("Error generating price comparison charts.")

        # Show the power plots
        fig_day_ahead_power, fig_imbalance_power = plot_power(day_ahead_data, imbalance_data)
        st.write('### Power Usage:')
        st.plotly_chart(fig_day_ahead_power)
        st.plotly_chart(fig_imbalance_power)

if __name__ == '__main__':
    main()


