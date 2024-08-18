import pandas as pd
import streamlit as st
from entsoe import EntsoePandasClient
from decimal import Decimal
from scipy.signal import find_peaks
import plotly.graph_objs as go

# required own api token from entsoe
API_TOKEN = '0464a296-1b5d-4be6-a037-b3414de630f8'
client = EntsoePandasClient(api_key=API_TOKEN)
st.title('Boiler Efficiency and Power Analysis Tool')
st.markdown("""How it works?

This tool allows you to compare the efficiency and costs between E-boilers and Gas-boilers based on day-ahead and imbalance electricity prices.
You can select the date range, country, gas price, and desired power that you wish to get out of these boilers to analyze the costs and determine which boiler is more cost-effective.
The results are displayed in interactive plots, and a summary of the key findings is provided.
""")

# this function gets the day-ahead prices from entsoe
def get_day_ahead_data(start, end, country_code):
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
    
    if 'index' in imbalance_prices.columns:
        imbalance_prices.rename(columns={'index': 'Time'}, inplace=True)
    else:
        st.error("The time column was not found in the imbalance prices data")
        return pd.DataFrame()  
    
    if 'Long' in imbalance_prices.columns and 'Short' in imbalance_prices.columns:
        imbalance_prices['Imbalance_Price_EUR_per_MWh'] = imbalance_prices[['Long', 'Short']].mean(axis=1)
    else:
        st.error("The expected two imbalance prices columns are not found")
        return pd.DataFrame()
    
    imbalance_prices = imbalance_prices[['Time', 'Imbalance_Price_EUR_per_MWh']]
    
    return imbalance_prices


# Function to determine the efficient boiler for day-ahead data
def efficient_boiler_day_ahead(day_ahead_price, gas_price):
    if pd.isna(day_ahead_price):
        return 'Unknown'
    if Decimal(day_ahead_price) < Decimal(gas_price) / (1000):
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# Function to determine the efficient boiler for imbalance data
def efficient_boiler_imbalance(imbalance_price, gas_price):
    if pd.isna(imbalance_price):
        return 'Unknown'
    if Decimal(imbalance_price) < Decimal(gas_price) / 1000:
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# Function to calculate costs for day-ahead data
def calculate_costs_day_ahead(data, gas_price):
    data['Efficient_Boiler_Day_Ahead'] = data['Day-Ahead_Price_EUR_per_MWh'].apply(efficient_boiler_day_ahead, gas_price=gas_price)
    return data

# Function to calculate costs for imbalance data
def calculate_costs_imbalance(data, gas_price):
    data['Efficient_Boiler_Imbalance'] = data['Imbalance_Price_EUR_per_MWh'].apply(efficient_boiler_imbalance, gas_price=gas_price)
    return data

# Function to calculate power usage for day-ahead data
def calculate_power_day_ahead(data, desired_power):
    data['E-boiler_Power_Day_Ahead'] = data.apply(lambda x: desired_power if x['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Day_Ahead'] = data.apply(lambda x: desired_power if x['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else 0, axis=1)
    return data

# Function to calculate power usage for imbalance data
def calculate_power_imbalance(data, desired_power):
    data['E-boiler_Power_Imbalance'] = data.apply(lambda x: desired_power if x['Efficient_Boiler_Imbalance'] == 'E-boiler' else 0, axis=1)
    data['Gas-boiler_Power_Imbalance'] = data.apply(lambda x: desired_power if x['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else 0, axis=1)
    return data

# Function to calculate savings for day-ahead data
def calculate_savings_day_ahead(data, gas_price, desired_power):
    # Convert everything to Decimal early on
    gas_price_mwh = (gas_price) * (1000)
    desired_power_mwh = (desired_power) / (1000)  # Convert kW to MWh
    
    # Calculate the cost for each time point directly
    data['Gas_Boiler_Cost_in_Euro'] = data.apply(lambda row: desired_power_mwh * gas_price_mwh 
                                         if row['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else (0), axis=1)
    
    # Sum the costs to get the total gas boiler cost
    gas_boiler_cost = data['Gas_Boiler_Cost_in_Euro'].sum()
    
    # Calculate the e-boiler cost similarly
    data['E_Boiler_Cost_in_Euro'] = data.apply(lambda row: desired_power_mwh * (row['Day-Ahead_Price_EUR_per_MWh']) 
                                       if row['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else (0), axis=1)
    e_boiler_cost = data['E_Boiler_Cost_in_Euro'].sum()
    
    # Calculate savings
    total_savings = (abs(e_boiler_cost))
    percentage_savings = ((total_savings) / (gas_boiler_cost) * (100)) if gas_boiler_cost else (0)
    
    # Return the calculated savings, percentages, and costs
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost

# Function to calculate savings for imbalance data
def calculate_savings_imbalance(data, gas_price, desired_power):
    # Convert everything to Decimal early on
    gas_price_mwh = (gas_price) * (1000)
    desired_power_mwh = (desired_power) / (1000)  # Convert kW to MWh
    
    # Calculate the cost for each time point directly
    data['Gas_Boiler_Cost_Imbalance_in_Euro'] = data.apply(lambda row: desired_power_mwh * gas_price_mwh 
                                                   if row['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else (0), axis=1)
    
    # Sum the costs to get the total gas boiler cost
    gas_boiler_cost = data['Gas_Boiler_Cost_Imbalance_in_Euro'].sum()
    
    # Calculate the e-boiler cost similarly
    data['E_Boiler_Cost_Imbalance_in_Euro'] = data.apply(lambda row: desired_power_mwh * (row['Imbalance_Price_EUR_per_MWh']) 
                                                 if row['Efficient_Boiler_Imbalance'] == 'E-boiler' else (0), axis=1)
    e_boiler_cost = data['E_Boiler_Cost_Imbalance_in_Euro'].sum()
    
    # Calculate savings
    total_savings = (abs(e_boiler_cost))
    percentage_savings = ((total_savings) / (gas_boiler_cost) * (100)) if gas_boiler_cost else (0)
    
    # Return the calculated savings, percentages, and costs
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost



def plot_price(day_ahead_data, imbalance_data):
    # Plot for Day-Ahead E-boiler and Gas-boiler costs
    day_ahead_fig = go.Figure()

    # Convert time to datetime
    day_ahead_data['Time'] = pd.to_datetime(day_ahead_data['Time'])

    # Plot day-ahead E-boiler and Gas-boiler costs
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['E_Boiler_Cost_in_Euro'],
                                       mode='lines', name='E-boiler Cost (Day-Ahead)', line=dict(color='blue')))
    day_ahead_fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['Gas_Boiler_Cost_in_Euro'],
                                       mode='lines', name='Gas-boiler Cost (Day-Ahead)', line=dict(color='red', dash='dash')))

    day_ahead_fig.update_layout(title='Day-Ahead E-boiler vs Gas-boiler Costs',
                                xaxis_title='Time',
                                yaxis_title='Cost (EUR)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    # Plot for Imbalance E-boiler and Gas-boiler costs
    imbalance_fig = go.Figure()

    # Convert time to datetime
    imbalance_data['Time'] = pd.to_datetime(imbalance_data['Time'])

    # Plot imbalance E-boiler and Gas-boiler costs
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['E_Boiler_Cost_Imbalance_in_Euro'],
                                       mode='lines', name='E-boiler Cost (Imbalance)', line=dict(color='blue')))
    imbalance_fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['Gas_Boiler_Cost_Imbalance_in_Euro'],
                                       mode='lines', name='Gas-boiler Cost (Imbalance)', line=dict(color='red', dash='dash')))

    imbalance_fig.update_layout(title='Imbalance E-boiler vs Gas-boiler Costs',
                                xaxis_title='Time',
                                yaxis_title='Cost (EUR)',
                                xaxis=dict(tickformat='%Y-%m-%d'),
                                legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    return day_ahead_fig, imbalance_fig

def plot_power(day_ahead_data, imbalance_data):
    # Calculate the sum of power for day-ahead data
    sum_e_boiler_day_ahead = day_ahead_data['E-boiler_Power_Day_Ahead'].sum()
    sum_gas_boiler_day_ahead = day_ahead_data['Gas-boiler_Power_Day_Ahead'].sum()

    # Create horizontal bar chart for day-ahead data using Plotly
    day_ahead_fig = go.Figure()
    day_ahead_fig.add_trace(go.Bar(
        y=['E-boiler (Day-Ahead)', 'Gas-boiler (Day-Ahead)'],
        x=[sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))

    day_ahead_fig.update_layout(title='Total Power - Day-Ahead',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_day_ahead, sum_gas_boiler_day_ahead) * 1.1]))

    # Calculate the sum of power for imbalance data
    sum_e_boiler_imbalance = imbalance_data['E-boiler_Power_Imbalance'].sum()
    sum_gas_boiler_imbalance = imbalance_data['Gas-boiler_Power_Imbalance'].sum()

    # Create horizontal bar chart for imbalance data using Plotly
    imbalance_fig = go.Figure()
    imbalance_fig.add_trace(go.Bar(
        y=['E-boiler (Imbalance)', 'Gas-boiler (Imbalance)'],
        x=[sum_e_boiler_imbalance, sum_gas_boiler_imbalance],
        orientation='h',
        marker=dict(color=['blue', 'red'])
    ))

    imbalance_fig.update_layout(title='Total Power - Imbalance',
                                xaxis_title='Total Power (kW)',
                                yaxis_title='',
                                xaxis=dict(range=[0, max(sum_e_boiler_imbalance, sum_gas_boiler_imbalance) * 1.1]))

    return day_ahead_fig, imbalance_fig



def main():

    
    # Sidebar settings for user input
    st.sidebar.title('Settings')
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2024-01-01'))
    country_code = st.sidebar.text_input('Country code', 'NL')
    gas_price = st.sidebar.number_input('Gas price per kWh', value=0.30 / 9.796)
    desired_power = st.sidebar.number_input('Desired Power (kW)', min_value=0.0, value=100.0, step=1.0)
    
    if st.sidebar.button('Get Data'):
        # Fetching data
        day_ahead_data = get_day_ahead_data(start_date, end_date, country_code)
        imbalance_data = get_imbalance_data(start_date, end_date, country_code)
        
        if day_ahead_data.empty or imbalance_data.empty:
            st.error("No data available")
        else:
            # Calculating costs and power usage for both day-ahead and imbalance data
            day_ahead_data = calculate_costs_day_ahead(day_ahead_data, gas_price)
            imbalance_data = calculate_costs_imbalance(imbalance_data, gas_price)
            
            day_ahead_data = calculate_power_day_ahead(day_ahead_data, desired_power)
            imbalance_data = calculate_power_imbalance(imbalance_data, desired_power)
            
            # Calculating savings for both day-ahead and imbalance data
            total_savings_day_ahead, percentage_savings_day_ahead, e_boiler_cost_day_ahead, gas_boiler_cost_day_ahead = calculate_savings_day_ahead(day_ahead_data, gas_price, desired_power)
            total_savings_imbalance, percentage_savings_imbalance, e_boiler_cost_imbalance, gas_boiler_cost_imbalance = calculate_savings_imbalance(imbalance_data, gas_price, desired_power)
            
            total_cost_day_ahead = (gas_boiler_cost_day_ahead) - (abs(e_boiler_cost_day_ahead))
            total_cost_imbalance = (gas_boiler_cost_imbalance) - (abs(e_boiler_cost_imbalance))
            
            # Displaying the results with wider columns or st.write() to avoid truncation
            st.write('### Day-Ahead Data Results:')
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 3, 3, 3, 3])
                col1.write(f"**Total Savings:**\n{total_savings_day_ahead:,.2f} EUR")
                col2.write(f"**Percentage Savings:**\n{percentage_savings_day_ahead:.2f}%")
                col3.write(f"**Total Cost:**\n{total_cost_day_ahead:,.2f} EUR")
                col4.write(f"**E-boiler Cost:**\n{e_boiler_cost_day_ahead:,.2f} EUR")
                col5.write(f"**Gas-boiler Cost:**\n{gas_boiler_cost_day_ahead:,.2f} EUR")

            st.write('### Imbalance Data Results:')
            with st.container():
                col6, col7, col8, col9, col10 = st.columns([3, 3, 3, 3, 3])
                col6.write(f"**Total Savings:**\n{total_savings_imbalance:,.2f} EUR")
                col7.write(f"**Percentage Savings:**\n{percentage_savings_imbalance:.2f}%")
                col8.write(f"**Total Cost:**\n{total_cost_imbalance:,.2f} EUR")
                col9.write(f"**E-boiler Cost:**\n{e_boiler_cost_imbalance:,.2f} EUR")
                col10.write(f"**Gas-boiler Cost:**\n{gas_boiler_cost_imbalance:,.2f} EUR")

            # Display the data tables
            st.write('### Day-Ahead Data Table:')
            st.dataframe(day_ahead_data)

            st.write('### Imbalance Data Table:')
            st.dataframe(imbalance_data)
            
            # Display the price plots
            fig_day_ahead_price, fig_imbalance_price = plot_price(day_ahead_data, imbalance_data)
            st.write('### Price Comparison:')
            st.plotly_chart(fig_day_ahead_price)
            st.plotly_chart(fig_imbalance_price)
            
            # Display the power plots
            fig_day_ahead_power, fig_imbalance_power = plot_power(day_ahead_data, imbalance_data)
            st.write('### Power Usage:')
            st.plotly_chart(fig_day_ahead_power)
            st.plotly_chart(fig_imbalance_power)

if __name__ == '__main__':
    main()
