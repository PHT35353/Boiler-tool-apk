import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import matplotlib.dates as mdates 
from entsoe import EntsoePandasClient
from scipy.signal import find_peaks
from decimal import Decimal

# required own api token from entsoe
API_TOKEN = '0464a296-1b5d-4be6-a037-b3414de630f8'
client = EntsoePandasClient(api_key=API_TOKEN)

# This function gets the day-ahead prices from entsoe
def get_day_ahead_data(start, end, country_code):
    start = pd.Timestamp(start, tz='Europe/Brussels')
    end = pd.Timestamp(end, tz='Europe/Brussels')
    
    # Getting day-ahead prices
    day_ahead_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    day_ahead_prices = day_ahead_prices.reset_index()
    day_ahead_prices.columns = ['Time', 'Day-Ahead_Price_EUR_per_MWh']
    
    return day_ahead_prices

# This function gets the imbalance prices from entsoe
def get_imbalance_data(start, end, country_code):
    start = pd.Timestamp(start, tz='Europe/Brussels')
    end = pd.Timestamp(end, tz='Europe/Brussels')
    
    # Getting imbalance prices
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
    if Decimal(day_ahead_price) < Decimal(gas_price) / Decimal(1000):
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# Function to determine the efficient boiler for imbalance data
def efficient_boiler_imbalance(imbalance_price, gas_price):
    if pd.isna(imbalance_price):
        return 'Unknown'
    if Decimal(imbalance_price) < Decimal(gas_price) / Decimal(1000):
        return 'E-boiler'
    else:
        return 'Gas-boiler'

# Function to calculate costs for day-ahead data
def calculate_costs_day_ahead(data, gas_price):
    data['Efficient_Boiler_Day_Ahead'] = data['Day-Ahead_Price_EUR_per_MWh'].apply(lambda x: efficient_boiler_day_ahead(x, gas_price))
    return data

# Function to calculate costs for imbalance data
def calculate_costs_imbalance(data, gas_price):
    data['Efficient_Boiler_Imbalance'] = data['Imbalance_Price_EUR_per_MWh'].apply(lambda x: efficient_boiler_imbalance(x, gas_price))
    return data

# Function to calculate power usage for day-ahead data
def calculate_power_day_ahead(data, desired_power):
    data['E-boiler_Power_Day_Ahead'] = data.apply(lambda x: Decimal(desired_power) if x['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else Decimal(0), axis=1)
    data['Gas-boiler_Power_Day_Ahead'] = data.apply(lambda x: Decimal(desired_power) if x['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else Decimal(0), axis=1)
    return data

# Function to calculate power usage for imbalance data
def calculate_power_imbalance(data, desired_power):
    data['E-boiler_Power_Imbalance'] = data.apply(lambda x: Decimal(desired_power) if x['Efficient_Boiler_Imbalance'] == 'E-boiler' else Decimal(0), axis=1)
    data['Gas-boiler_Power_Imbalance'] = data.apply(lambda x: Decimal(desired_power) if x['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else Decimal(0), axis=1)
    return data

# Function to calculate savings for day-ahead data
def calculate_savings_day_ahead(data, gas_price, desired_power):
    # Convert everything to Decimal early on
    gas_price_mwh = Decimal(gas_price) * Decimal(1000)
    desired_power_mwh = Decimal(desired_power) / Decimal(1000)  # Convert kW to MWh
    
    # Calculate the cost for each time point directly
    data['Gas_Boiler_Cost'] = data.apply(lambda row: desired_power_mwh * gas_price_mwh 
                                         if row['Efficient_Boiler_Day_Ahead'] == 'Gas-boiler' else Decimal(0), axis=1)
    
    # Sum the costs to get the total gas boiler cost
    gas_boiler_cost = data['Gas_Boiler_Cost'].sum()
    
    # Calculate the e-boiler cost similarly
    data['E_Boiler_Cost'] = data.apply(lambda row: desired_power_mwh * Decimal(row['Day-Ahead_Price_EUR_per_MWh']) 
                                       if row['Efficient_Boiler_Day_Ahead'] == 'E-boiler' else Decimal(0), axis=1)
    e_boiler_cost = data['E_Boiler_Cost'].sum()
    
    # Calculate savings
    total_savings = abs(e_boiler_cost)
    percentage_savings = (total_savings / gas_boiler_cost * Decimal(100)) if gas_boiler_cost else Decimal(0)
    
    # Return the calculated savings, percentages, and costs
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost

# Function to calculate savings for imbalance data
def calculate_savings_imbalance(data, gas_price, desired_power):
    # Convert everything to Decimal early on
    gas_price_mwh = Decimal(gas_price) * Decimal(1000)
    desired_power_mwh = Decimal(desired_power) / Decimal(1000)  # Convert kW to MWh
    
    # Calculate the cost for each time point directly
    data['Gas_Boiler_Cost_Imbalance'] = data.apply(lambda row: desired_power_mwh * gas_price_mwh 
                                                   if row['Efficient_Boiler_Imbalance'] == 'Gas-boiler' else Decimal(0), axis=1)
    
    # Sum the costs to get the total gas boiler cost
    gas_boiler_cost = data['Gas_Boiler_Cost_Imbalance'].sum()
    
    # Calculate the e-boiler cost similarly
    data['E_Boiler_Cost_Imbalance'] = data.apply(lambda row: desired_power_mwh * Decimal(row['Imbalance_Price_EUR_per_MWh']) 
                                                 if row['Efficient_Boiler_Imbalance'] == 'E-boiler' else Decimal(0), axis=1)
    e_boiler_cost = data['E_Boiler_Cost_Imbalance'].sum()
    
    # Calculate savings
    total_savings = abs(e_boiler_cost)
    percentage_savings = (total_savings / gas_boiler_cost * Decimal(100)) if gas_boiler_cost else Decimal(0)
    
    # Return the calculated savings, percentages, and costs
    return total_savings, percentage_savings, e_boiler_cost, gas_boiler_cost




def plot_price(day_ahead_data, imbalance_data, gas_price):
    fig, ax = plt.subplots(figsize=(12, 6)) 

    # Convert time to datetime and set as index
    day_ahead_data['Time'] = pd.to_datetime(day_ahead_data['Time'])
    day_ahead_data.set_index('Time', inplace=True)
    
    imbalance_data['Time'] = pd.to_datetime(imbalance_data['Time'])
    imbalance_data.set_index('Time', inplace=True)
    
    # Plot day-ahead prices
    ax.plot(day_ahead_data.index, day_ahead_data['Day-Ahead_Price_EUR_per_MWh'], color='red', label='Day-Ahead E-boiler Price', linewidth=0.5, alpha=0.7)
    
    # Plot imbalance prices
    ax.plot(imbalance_data.index, imbalance_data['Imbalance_Price_EUR_per_MWh'], color='blue', label='Imbalance E-boiler Price', linewidth=0.5, alpha=0.7)

    # Plot gas price as a constant line
    ax.axhline(y=gas_price * 1000, color='green', linestyle='--', label='Gas Price (EUR/MWh)', linewidth=1)

    ax.set_title('Boiler Price Efficiency Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price EUR per MWh')
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


# this function is for plotting the power graph


def plot_power(day_ahead_data, imbalance_data):
    fig, ax = plt.subplots(figsize=(14, 8))  

    # Plot peaks and zero points for day-ahead prices
    e_boiler_peaks_day_ahead, _ = find_peaks(day_ahead_data['E-boiler_Power_Day_Ahead'], distance=5, prominence=1)
    gas_boiler_peaks_day_ahead, _ = find_peaks(day_ahead_data['Gas-boiler_Power_Day_Ahead'], distance=5, prominence=1)

    e_boiler_zeros_day_ahead = day_ahead_data[day_ahead_data['E-boiler_Power_Day_Ahead'] == 0].index
    gas_boiler_zeros_day_ahead = day_ahead_data[day_ahead_data['Gas-boiler_Power_Day_Ahead'] == 0].index

    ax.plot(day_ahead_data.index[e_boiler_peaks_day_ahead], day_ahead_data['E-boiler_Power_Day_Ahead'].iloc[e_boiler_peaks_day_ahead], 'b^-', label='Day-Ahead E-boiler Peaks', markersize=6)
    ax.plot(e_boiler_zeros_day_ahead, day_ahead_data['E-boiler_Power_Day_Ahead'].loc[e_boiler_zeros_day_ahead], 'b^', label='Day-Ahead E-boiler Zeros', markersize=6)
    ax.plot(day_ahead_data.index[gas_boiler_peaks_day_ahead], day_ahead_data['Gas-boiler_Power_Day_Ahead'].iloc[gas_boiler_peaks_day_ahead], 'r^-', label='Day-Ahead Gas-boiler Peaks', markersize=6)
    ax.plot(gas_boiler_zeros_day_ahead, day_ahead_data['Gas-boiler_Power_Day_Ahead'].loc[gas_boiler_zeros_day_ahead], 'r^', label='Day-Ahead Gas-boiler Zeros', markersize=6)

    # Plot peaks and zero points for imbalance prices
    e_boiler_peaks_imbalance, _ = find_peaks(imbalance_data['E-boiler_Power_Imbalance'], distance=5, prominence=1)
    gas_boiler_peaks_imbalance, _ = find_peaks(imbalance_data['Gas-boiler_Power_Imbalance'], distance=5, prominence=1)

    e_boiler_zeros_imbalance = imbalance_data[imbalance_data['E-boiler_Power_Imbalance'] == 0].index
    gas_boiler_zeros_imbalance = imbalance_data[imbalance_data['Gas-boiler_Power_Imbalance'] == 0].index

    ax.plot(imbalance_data.index[e_boiler_peaks_imbalance], imbalance_data['E-boiler_Power_Imbalance'].iloc[e_boiler_peaks_imbalance], 'g^-', label='Imbalance E-boiler Peaks', markersize=6)
    ax.plot(e_boiler_zeros_imbalance, imbalance_data['E-boiler_Power_Imbalance'].loc[e_boiler_zeros_imbalance], 'g^', label='Imbalance E-boiler Zeros', markersize=6)
    ax.plot(imbalance_data.index[gas_boiler_peaks_imbalance], imbalance_data['Gas-boiler_Power_Imbalance'].iloc[gas_boiler_peaks_imbalance], 'y^-', label='Imbalance Gas-boiler Peaks', markersize=6)
    ax.plot(gas_boiler_zeros_imbalance, imbalance_data['Gas-boiler_Power_Imbalance'].loc[gas_boiler_zeros_imbalance], 'y^', label='Imbalance Gas-boiler Zeros', markersize=6)

    ax.set_title('Boiler Power Delivery - Peaks and Zeros')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (kW)')
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def main():
    st.title('Boiler Efficiency and Power Analysis')
    
    # Sidebar settings for user input
    st.sidebar.title('Settings')
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2024-01-01'))
    country_code = st.sidebar.text_input('Country code', 'NL')
    gas_price = st.sidebar.number_input('Gas price per kWh', value= 0.30/9.796)
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
            
            total_cost_day_ahead = Decimal(gas_boiler_cost_day_ahead) - Decimal(abs(e_boiler_cost_day_ahead))
            total_cost_imbalance = Decimal(gas_boiler_cost_imbalance) - Decimal(abs(e_boiler_cost_imbalance))
            
            # Displaying the results in Streamlit
            st.write('### Day-Ahead Data Results:')
            st.write(f'Total Savings (Day-Ahead): {total_savings_day_ahead:.2f} EUR')
            st.write(f'Percentage Savings (Day-Ahead): {percentage_savings_day_ahead:.2f}%')
            st.write(f'E-boiler Cost (Day-Ahead): {e_boiler_cost_day_ahead:.2f} EUR')
            st.write(f'Gas-boiler Cost (Day-Ahead): {gas_boiler_cost_day_ahead:.2f} EUR')
            st.write(f'Total Cost (Day-Ahead): {total_cost_day_ahead:.2f} EUR')
            
            st.write('### Imbalance Data Results:')
            st.write(f'Total Savings (Imbalance): {total_savings_imbalance:.2f} EUR')
            st.write(f'Percentage Savings (Imbalance): {percentage_savings_imbalance:.2f}%')
            st.write(f'E-boiler Cost (Imbalance): {e_boiler_cost_imbalance:.2f} EUR')
            st.write(f'Gas-boiler Cost (Imbalance): {gas_boiler_cost_imbalance:.2f} EUR')
            st.write(f'Total Cost (Imbalance): {total_cost_imbalance:.2f} EUR')

            # Display the data tables
            st.write('### Day-Ahead Data Table:')
            st.dataframe(day_ahead_data)

            st.write('### Imbalance Data Table:')
            st.dataframe(imbalance_data)
            
            # Displaying the plots
            price_fig = plot_price(day_ahead_data, imbalance_data, gas_price)
            if price_fig:
                st.pyplot(price_fig)
            
            power_fig = plot_power(day_ahead_data, imbalance_data)
            if power_fig:
                st.pyplot(power_fig)

if __name__ == '__main__':
    main()
