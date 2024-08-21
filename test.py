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


# this function plots the price graph of day-ahead and imbalance data with plotly

    def plot_price(day_ahead_data, imbalance_data, gas_price):
    # Convert E-boiler costs back to EUR/kWh for both day-ahead and imbalance markets
    day_ahead_data['E_Boiler_Price_EUR_per_kWh'] = day_ahead_data['E_Boiler_Cost_in_Euro'] / (day_ahead_data['Desired Power'] / 1000)
    imbalance_data['E_Boiler_Price_EUR_per_kWh'] = imbalance_data['E_Boiler_Cost_Imbalance_in_Euro'] / (imbalance_data['Desired Power'] / 1000)
    
    # Gas price in EUR/kWh remains constant
    gas_boiler_price_kWh = gas_price

    # Plotting the Day-Ahead and Imbalance prices along with the Gas-boiler price
    fig = go.Figure()

    # Day-Ahead E-boiler price
    fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=day_ahead_data['E_Boiler_Price_EUR_per_kWh'],
                             mode='lines', name='E-boiler Price (Day-Ahead)', line=dict(color='blue')))
    
    # Imbalance E-boiler price
    fig.add_trace(go.Scatter(x=imbalance_data['Time'], y=imbalance_data['E_Boiler_Price_EUR_per_kWh'],
                             mode='lines', name='E-boiler Price (Imbalance)', line=dict(color='green')))
    
    # Constant Gas-boiler price
    fig.add_trace(go.Scatter(x=day_ahead_data['Time'], y=[gas_boiler_price_kWh]*len(day_ahead_data),
                             mode='lines', name='Gas-boiler Price', line=dict(color='red', dash='dash')))

    # Adding titles and axis labels
    fig.update_layout(title='E-boiler and Gas-boiler Prices Comparison',
                      xaxis_title='Time',
                      yaxis_title='Price (EUR/kWh)',
                      xaxis=dict(tickformat='%Y-%m-%d'),
                      legend=dict(x=0, y=-0.2, xanchor='left', yanchor='top'))

    return fig

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
    # This function makes the sidebar of settings
    st.sidebar.title('Settings')
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2024-01-01'))
    country_code = st.sidebar.text_input('Country code', 'NL')
    gas_price = st.sidebar.number_input('Gas price EUR/kWh', value=0.30 / 9.796)
    desired_power = st.sidebar.number_input('Desired Power (kWh)', min_value=0.0, value=100.0, step=1.0)
    uploaded_file = st.sidebar.file_uploader("Upload your desired power data (Excel file)", type=["xlsx", "xls"])

    if st.sidebar.button('Get Data'):
        day_ahead_data = get_day_ahead_data(start_date, end_date, country_code)
        imbalance_data = get_imbalance_data(start_date, end_date, country_code)

        if day_ahead_data.empty or imbalance_data.empty:
            st.error("No data available")
        else:
            # Process uploaded file if available, otherwise use the desired power input
            if uploaded_file is not None:
                try:
                    # Attempt to read the Excel file
                    uploaded_data = pd.read_excel(uploaded_file)

                    # Ensure the file has the correct structure (e.g., 'Time' and 'Desired Power' columns)
                    if 'Time' in uploaded_data.columns and 'Desired Power' in uploaded_data.columns:
                        uploaded_data['Time'] = pd.to_datetime(uploaded_data['Time'])
                        # Merge with day-ahead and imbalance data based on time
                        day_ahead_data = pd.merge(day_ahead_data, uploaded_data[['Time', 'Desired Power']], on='Time', how='left')
                        imbalance_data = pd.merge(imbalance_data, uploaded_data[['Time', 'Desired Power']], on='Time', how='left')
                        # Use the uploaded 'Desired Power' data
                        day_ahead_data['Desired Power'] = day_ahead_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')
                        imbalance_data['Desired Power'] = imbalance_data['Desired Power'].fillna(method='ffill').fillna(method='bfill')
                    else:
                        st.error("Uploaded file must contain 'Time' and 'Desired Power' columns")
                        return
                except Exception as e:
                    st.error(f"Error reading the uploaded file: {str(e)}")
                    return
            else:
                day_ahead_data['Desired Power'] = desired_power
                imbalance_data['Desired Power'] = desired_power

            # Calculate the costs and power usage for both day-ahead and imbalance data
            day_ahead_data = day_ahead_costs(day_ahead_data, gas_price)
            imbalance_data = imbalance_costs(imbalance_data, gas_price)

            day_ahead_data = day_ahead_power(day_ahead_data)
            imbalance_data = imbalance_power(imbalance_data)

            # Calculate savings for both day-ahead and imbalance data
            total_savings_day_ahead, percentage_savings_day_ahead, e_boiler_cost_day_ahead, gas_boiler_cost_day_ahead = calculate_savings_day_ahead(day_ahead_data, gas_price, desired_power)
            total_savings_imbalance, percentage_savings_imbalance, e_boiler_cost_imbalance, gas_boiler_cost_imbalance, imbalance_data = calculate_savings_imbalance(imbalance_data, gas_price, desired_power)

            total_cost_day_ahead = gas_boiler_cost_day_ahead - abs(e_boiler_cost_day_ahead)
            total_cost_imbalance = gas_boiler_cost_imbalance - abs(e_boiler_cost_imbalance)

            # Drop the 'Time_Diff_Minutes' column before displaying
            imbalance_data_display = imbalance_data.drop(columns=['Time_Diff_Minutes'])

            # Display the results
            st.write('### Day-Ahead Data Results:')
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([10, 10, 10, 10, 10])
                col1.write(f"**Total Savings:**\n{total_savings_day_ahead:,.2f} EUR")
                col2.write(f"**Percentage Savings:**\n{percentage_savings_day_ahead:.2f}%")
                col3.write(f"**Total Cost:**\n{total_cost_day_ahead:,.2f} EUR")
                col4.write(f"**E-boiler Cost:**\n{e_boiler_cost_day_ahead:,.2f} EUR")
                col5.write(f"**Gas-boiler Cost:**\n{gas_boiler_cost_day_ahead:,.2f} EUR")

            st.write('### Imbalance Data Results:')
            with st.container():
                col6, col7, col8, col9, col10 = st.columns([10, 10, 10, 10, 10])
                col6.write(f"**Total Savings:**\n{total_savings_imbalance:,.2f} EUR")
                col7.write(f"**Percentage Savings:**\n{percentage_savings_imbalance:.2f}%")
                col8.write(f"**Total Cost:**\n{total_cost_imbalance:,.2f} EUR")
                col9.write(f"**E-boiler Cost:**\n{e_boiler_cost_imbalance:,.2f} EUR")
                col10.write(f"**Gas-boiler Cost:**\n{gas_boiler_cost_imbalance:,.2f} EUR")

            # Show the data tables
            st.write('### Day-Ahead Data Table:')
            st.dataframe(day_ahead_data)

            st.write('### Imbalance Data Table:')
            st.dataframe(imbalance_data_display)

            # Show the price plots
            fig_day_ahead_price, fig_imbalance_price = plot_price(day_ahead_data, imbalance_data_display)
            st.write('### Price Comparison:')
            st.plotly_chart(fig_day_ahead_price)
            st.plotly_chart(fig_imbalance_price)

            # Show the power plots
            fig_day_ahead_power, fig_imbalance_power = plot_power(day_ahead_data, imbalance_data_display)
            st.write('### Power Usage:')
            st.plotly_chart(fig_day_ahead_power)
            st.plotly_chart(fig_imbalance_power)

if __name__ == '__main__':
    main()
