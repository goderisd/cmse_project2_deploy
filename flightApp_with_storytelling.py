import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf



def load_airline_data(airline_name):
    base_path = os.path.dirname(__file__)

    if airline_name == 'Delta':
        non_cancelled_path = os.path.join(base_path, 'downsampled_merged_weather_holiday_non_cancelled_delta_flights.csv')
        cancelled_path = os.path.join(base_path, 'downsampled_merged_weather_holiday_canceled_delta_flights.csv')
    elif airline_name == 'American':
        non_cancelled_path = os.path.join(base_path,'downsampled_merged_weather_holiday_non_cancelled_american_flights.csv')
        cancelled_path = os.path.join(base_path,'downsampled_merged_weather_holiday_canceled_american_flights.csv')
    elif airline_name == 'United':
        non_cancelled_path = os.path.join(base_path, 'downsampled_merged_weather_holiday_non_cancelled_united_flights.csv')
        cancelled_path = os.path.join(base_path,'downsampled_merged_weather_holiday_canceled_united_flights.csv')

    non_cancelled_data = pd.read_csv(non_cancelled_path, parse_dates=['FL_DATE'])
    cancelled_data = pd.read_csv(cancelled_path, parse_dates=['FL_DATE'])

    return non_cancelled_data, cancelled_data


# Home page
def homepage():

    st.subheader("Introduction")
    st.write("""
        **Who:** Travelers, airline operators, and data analysts.
        **What:** Analyzes and predicts flight delays due to weather and holidays.
        **When:** Using data from 2019 to 2023, focusing on 2022.
        **Where:** U.S. airports, for Delta, American, and United Airlines.
        **Why:** Helps passengers and airlines mitigate delay risks.
        **How:** Integrates datasets for flight, weather, and federal holidays to provide insights.
    """)
    st.title("Airline Delay Analysis")
    st.write("""
        Welcome to the Airline Delay Prediction App! 
        This app allows you to explore how weather and holiday factors affect the major U.S. airlines: Delta, American Airlines, and United.
        
        This app uses 3 primary data sources:
         1. A USDOT flight dataset for all US carriers from 2019-2023
         2. A Weather events dataset from 2016-2022, focused on events local to zipcodes that contain US airports 
         3. A Federal holidays dataset 
         
        In the initial phase of development of the app, the dataset was down-sampled to only the 3 major US carriers, and also to only the year 2022, where the weather and the lfight data set overlapped.
        
        To handle the missing values in the datasets KNN was used on the individual airlines datasets, as well as the weather dataset.
        Once the datasets were whole each airline dataset was merged with a weather dataset based on time and airport code.
        This meant that a script looked at the airport code where the weather event occurred, and aligned it to the scheduled data and time for both the arrival and departure airport.
        This allows the app to look at the affect of both the weather conditions at the arrival and departure airports at the same time, to see how they impact flight delays.
        Finally, the last merge for the dataset was including data for federal holidays, using a binary encoding, the app looks at whether or not the presence of a federal holiday impacts flgiht delays.
        

        ## App Features:
        1. **Weather and Flight Data Integration**: View the impact of weather conditions on delays and cancellations.
        2. **Holiday Schedule Context**: See how federal holidays influence flight disruptions.
        3. **Interactive Visualizations**: Explore various visualizations, including delay trends, weather patterns, and more.
        4. **Prediction Models**: Predict delays and cancellations based on weather conditions and holidays.
        5. **Prediction Model Assessment**: See how your model parameters compare in accuracy to each other.

        ## Instructions:
        - Use the tabs to navigate between the major airlines.
        - Explore visualizations and insights specific to each airline.
        - Each page begins with visualaztions such as correlation heatmaps, impact of weather on flight delay for city pairs, impact of federal holidays on the specific airlines and more.
        - Next you can also input specific parameters to predict flight delays and see how your choices impact the delay projection.
        - Future enhancements to improve the prediction performance and consumer viability can be found at the base of each page.
        
        ## WARNING:
        - This is a sampled data set to allow for streamlit to be able to host this massive dataset 
        - This means that a few things may struggle to be accurate as the code random selects two days per month to display flight data 
        -Importantly the "Holiday projection" and visualization will likely be very inaccurate, since the data is sampled to two random days per month, meanign it will either be a holiday or not.
        - Furthermore, this means that the already poor ability to predict flight delays gets a lot worse :), if you would like a copy of an app that can run on your local machien with the full 2022 datset email me: goderisd@msu.edu.  I will be happy to send you the repo!
    """)


def plot_correlation_matrix(df):
    corr = df[['DEP_DELAY', 'Precipitation(in)_origin', 'Precipitation(in)_dest', 'Severity_origin_numeric',
               'Severity_dest_numeric']].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()


def plot_severity_origin_dep_delay(df):
    # Map severity to numeric values
    severity_map = {
        'Light': 1,
        'Moderate': 2,
        'Heavy': 3,
    }
    df['Severity_origin_numeric'] = df['Severity_origin'].map(severity_map)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Severity_origin_numeric', y='DEP_DELAY')  # Use DEP_DELAY for departure delays
    plt.title('Flight Departure Delay by Weather Severity at Origin')
    plt.xlabel('Severity at Origin (Numeric)')
    plt.ylabel('Departure Delay (Minutes)')
    st.pyplot()


def plot_severity_dest_arr_delay(df):
    # Map severity to numeric values
    severity_map = {
        'Light': 1,
        'Moderate': 2,
        'Heavy': 3,
    }
    df['Severity_dest_numeric'] = df['Severity_dest'].map(severity_map)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Severity_dest_numeric', y='ARR_DELAY')  # Use ARR_DELAY for arrival delays
    plt.title('Flight Arrival Delay by Weather Severity at Destination')
    plt.xlabel('Severity at Destination (Numeric)')
    plt.ylabel('Arrival Delay (Minutes)')
    st.pyplot()


def plot_delay_by_holiday(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Holiday_origin', y='DEP_DELAY', estimator='mean')
    plt.title('Average Departure Delay on Holiday vs. Non-Holiday')
    plt.xlabel('Holiday Status (Origin, 0 is no Holiday 1 is Holiday)')
    plt.ylabel('Average Departure Delay (Minutes)')
    st.pyplot()


def plot_delay_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['DEP_DELAY'], bins=50, kde=True, color='blue')  # Use DEP_DELAY for departure delays
    plt.title('Departure Delay Distribution')
    plt.xlabel('Departure Delay (Minutes)')
    plt.ylabel('Frequency')
    st.pyplot()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['ARR_DELAY'], bins=50, kde=True, color='orange')  # Use ARR_DELAY for arrival delays
    plt.title('Arrival Delay Distribution')
    plt.xlabel('Arrival Delay (Minutes)')
    plt.ylabel('Frequency')
    st.pyplot()


def plot_avg_delay_over_time(df):
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # Plotting average departure delay over time
    avg_delay_dep = df.groupby('FL_DATE')['DEP_DELAY'].mean()  # Use DEP_DELAY for average departure delays
    plt.figure(figsize=(10, 6))
    avg_delay_dep.plot(kind='line', color='green')
    plt.title('Average Departure Delay Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Departure Delay (minutes)')
    st.pyplot()


    avg_delay_arr = df.groupby('FL_DATE')['ARR_DELAY'].mean()  # Use ARR_DELAY for average arrival delays
    plt.figure(figsize=(10, 6))
    avg_delay_arr.plot(kind='line', color='red')
    plt.title('Average Arrival Delay Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Arrival Delay (minutes)')
    st.pyplot()


def plot_weather_impact_by_airports(df):

    st.subheader("Weather Impact Analysis")
    st.write("""
        **What:** This visualization shows how weather impacts delays across selected airports.
        **Why:** Understanding these impacts can help optimize operations at key airports.
        **How:** Uses precipitation data and departure/arrival delays.
    """)
            # Create multiselect dropdowns for airport selection
    st.sidebar.header("Select Airports")
    origin_airports = st.sidebar.multiselect("Select Origin Airports", df['ORIGIN_CITY'].unique())
    destination_airports = st.sidebar.multiselect("Select Destination Airports", df['DEST_CITY'].unique())

    # Filter the data based on selected airports
    filtered_df = df[(df['ORIGIN_CITY'].isin(origin_airports)) & (df['DEST_CITY'].isin(destination_airports))]

    # If no airports are selected, show a message
    if len(filtered_df) == 0:
        st.warning("No data available for the selected airports.")
        return

    # Plot for Departure Delays
    fig_dep = px.scatter(filtered_df, x='Precipitation(in)_origin', y='DEP_DELAY',
                         color='ORIGIN_CITY', symbol='DEST_CITY',
                         title=f'Weather Impact on Departure Delays from {", ".join(origin_airports)} to {", ".join(destination_airports)}',
                         labels={'Precipitation(in)_origin': 'Precipitation (inches)',
                                 'DEP_DELAY': 'Departure Delay (minutes)'},
                         hover_data=['ORIGIN_CITY', 'DEST_CITY'])  # Customize hover data
    fig_dep.update_layout(legend_title="Cities", showlegend=True)
    fig_dep.update_traces(marker=dict(size=10))  # Adjust marker size if needed
    st.plotly_chart(fig_dep)

    # Plot for Arrival Delays
    fig_arr = px.scatter(filtered_df, x='Precipitation(in)_dest', y='ARR_DELAY',
                         color='ORIGIN_CITY', symbol='DEST_CITY',
                         title=f'Weather Impact on Arrival Delays from {", ".join(origin_airports)} to {", ".join(destination_airports)}',
                         labels={'Precipitation(in)_dest': 'Precipitation (inches)',
                                 'ARR_DELAY': 'Arrival Delay (minutes)'},
                         hover_data=['ORIGIN_CITY', 'DEST_CITY'])  # Customize hover data
    fig_arr.update_layout(legend_title="Cities", showlegend=True)
    fig_arr.update_traces(marker=dict(size=10))  # Adjust marker size if needed
    st.plotly_chart(fig_arr)


def forecast_ar_model(df_forecast, forecast_type='Mean', p_value=1):
    # Ensure 'FL_DATE' is set as the index and sort by date
    df_forecast = df_forecast.sort_values(by='FL_DATE').set_index('FL_DATE')

    df_forecast = df_forecast.fillna(0)

    if forecast_type == 'Mean':
        # Mean forecast: simply use the mean of ARR_DELAY
        forecast = df_forecast['ARR_DELAY'].mean()
        return forecast, None, None, None  # No predictions or train/test data

    elif forecast_type == 'Weather Features':
        # Ensure relevant weather features exist in the dataframe
        weather_columns = ['Precipitation(in)_origin', 'Precipitation(in)_dest',
                           'Severity_origin_numeric', 'Severity_dest_numeric']
        df_forecast = df_forecast[weather_columns + ['ARR_DELAY']]

        # Split data into train and test (80-20 split)
        train_size = int(len(df_forecast) * 0.8)
        train_data = df_forecast.iloc[:train_size]
        test_data = df_forecast.iloc[train_size:]

        # Fit AR(p) Model on 'ARR_DELAY'
        model_ar = AutoReg(train_data['ARR_DELAY'], lags=p_value).fit()

        # Multi-step predictions on test data
        predictions = model_ar.predict(start=len(train_data), end=len(df_forecast) - 1, dynamic=False)

        # Forecast for the next step
        forecast = model_ar.forecast(steps=1)[0]
        return forecast, predictions, train_data, test_data

    elif forecast_type == 'Holiday':
        # Ensure 'Holiday_origin' is present
        if 'Holiday_origin' not in df_forecast.columns:
            raise ValueError("Holiday feature 'Holiday_origin' is not found in the dataframe.")

        # Add the 'Holiday' column as a numeric feature
        df_forecast['Holiday'] = df_forecast['Holiday_origin'].astype(int)

        # Split data into train and test (80-20 split)
        train_size = int(len(df_forecast) * 0.8)
        train_data = df_forecast.iloc[:train_size]
        test_data = df_forecast.iloc[train_size:]

        # Fit AR(p) Model on 'ARR_DELAY' (AR does not handle additional features directly)
        model_ar = AutoReg(train_data['ARR_DELAY'], lags=p_value).fit()

        # Multi-step predictions on test data
        predictions = model_ar.predict(start=len(train_data), end=len(df_forecast) - 1, dynamic=False)

        # Forecast for the next step
        forecast = model_ar.forecast(steps=1)[0]
        return forecast, predictions, train_data, test_data


    else:
        raise ValueError("Invalid forecast_type. Choose 'Mean', 'Weather Features', or 'Holiday'.")


def evaluate_model(true_values, predicted_values):
    # Calculate RMSE (Root Mean Squared Error) for model evaluation
    rmse_value = np.sqrt(mean_squared_error(true_values, predicted_values))
    return rmse_value


# Function to fit AR(p) model
def fit_ar_model(train_data, p, forecast_steps):
    # Create lagged matrix A and target vector b
    A = np.column_stack([train_data[i:len(train_data) - p + i] for i in range(p)])
    b = train_data[p:]

    # Solve for coefficients using least squares
    phi = np.linalg.solve(A.T @ A, A.T @ b)

    # Generate forecast
    forecast = []
    input_data = list(train_data[-p:])  # Start with the last p points
    for _ in range(forecast_steps):
        next_value = sum(phi[i] * input_data[-i - 1] for i in range(p))
        forecast.append(next_value)
        input_data.append(next_value)  # Append forecasted value for next prediction

    return np.array(forecast)


# AR Model Forecasting Function
def forecast_ar_model_with_mean(df_forecast, p_value=1, forecast_steps=1):

    # Ensure 'FL_DATE' is datetime and sort by date
    df_forecast['FL_DATE'] = pd.to_datetime(df_forecast['FL_DATE'])
    df_forecast = df_forecast.sort_values(by='FL_DATE')

    # Aggregate the mean ARR_DELAY by date
    mean_delay = df_forecast.groupby('FL_DATE')['ARR_DELAY'].mean()

    # Split data into train and test (80-20 split)
    train_size = int(len(mean_delay) * 0.8)
    train_data = mean_delay.iloc[:train_size]
    test_data = mean_delay.iloc[train_size:]

    # Fit AR(p) Model on aggregated mean delays
    model_ar = AutoReg(train_data, lags=p_value).fit()

    # Multi-step predictions on test data
    predictions = model_ar.predict(start=len(train_data), end=len(mean_delay) - 1, dynamic=False)

    # Forecast for the next steps
    forecast = model_ar.forecast(steps=forecast_steps)
    return forecast, predictions, train_data, test_data


# Plotting the results
def plot_ar_model_with_mean(train_data, test_data, predictions, forecast, p_value):

    # Create two subplots: one for the full plot and one for the zoomed-in version
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # Full plot (top)
    ax[0].plot(train_data.index, train_data, label="Training Data", color="blue")
    ax[0].plot(test_data.index, test_data, label="Actual Test Data", color="orange")
    ax[0].plot(test_data.index, predictions, label="Predictions", linestyle="--", color="green")
    ax[0].set_title(f"AR({p_value}) Forecast vs Actual on Aggregated Data")
    ax[0].set_ylabel("Mean Arrival Delay")
    ax[0].legend()

    # Zoomed-in plot (bottom)
    ax[1].plot(test_data.index[-len(predictions):], test_data[-len(predictions):], label="Actual Test Data",
               color="orange")
    ax[1].plot(test_data.index[-len(predictions):], predictions, label="Predictions", linestyle="--", color="green")
    ax[1].set_title("Zoomed-in Forecast vs Actual on Aggregated Data")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Mean Arrival Delay")
    ax[1].legend()

    # Add gridlines to both plots
    for axis in ax:
        axis.grid(True, linestyle="--", alpha=0.7)

    # Show the plots
    st.pyplot(fig)


def forecast_var_model(df, p_value):

    # Ensure FL_DATE is datetime
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

    # Aggregate daily averages
    aggregated = df.groupby('FL_DATE').agg({
        'ARR_DELAY': 'mean',
        'DEP_DELAY': 'mean',
        'AIR_TIME': 'mean'
    }).dropna()

    # Train-test split
    train_size = int(len(aggregated) * 0.8)
    train_data = aggregated.iloc[:train_size]
    test_data = aggregated.iloc[train_size:]

    # Fit VAR model
    model = VAR(train_data)
    results = model.fit(p_value)

    # Forecast
    forecast_steps = len(test_data)
    forecast = results.forecast(train_data.values[-p_value:], steps=forecast_steps)

    return forecast, results, train_data, test_data


def plot_var_results(train_data, test_data, forecast, p_value):
    # Convert forecast to DataFrame for plotting
    forecast_df = pd.DataFrame(
        forecast,
        index=test_data.index,
        columns=test_data.columns
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for i, col in enumerate(test_data.columns):
        axes[i].plot(train_data.index, train_data[col], label="Training Data", color="blue")
        axes[i].plot(test_data.index, test_data[col], label="Actual Test Data", color="orange")
        axes[i].plot(forecast_df.index, forecast_df[col], label="Forecast", linestyle="--", color="green")
        axes[i].set_title(f"VAR({p_value}) - {col}")
        axes[i].legend()
        axes[i].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

def evaluate_and_plot_statistics(model_name, true_values, predicted_values):
    min_length = min(len(true_values), len(predicted_values))
    true_values = true_values[:min_length]
    predicted_values = predicted_values[:min_length]

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = np.mean(np.abs(true_values - predicted_values))
    mbe = np.mean(true_values - predicted_values)  # Mean Bias Error

    # Display metrics in Streamlit
    st.subheader(f"Model: {model_name}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MBE: {mbe:.2f}")

    # Residuals
    residuals = true_values - predicted_values

    # Create subplots for each model's statistics
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    # Forecast vs Actual (Full)
    ax[0].plot(true_values.index, true_values, label="Actual Test Data", color="orange")
    ax[0].plot(true_values.index, predicted_values, label="Forecast", linestyle="--", color="green")
    ax[0].set_title(f"{model_name} Forecast vs Actual")
    ax[0].set_ylabel("Arrival Delay")
    ax[0].legend()

    ax[1].plot(true_values.index, residuals, label="Residuals (Forecast Error)", color="purple")
    ax[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ax[1].set_title(f"{model_name} Residuals (Forecast Error)")
    ax[1].set_ylabel("Error")
    ax[1].legend()

    ax[2].hist(residuals, bins=20, color="purple", edgecolor="black", alpha=0.7)
    ax[2].set_title(f"{model_name} Residuals Distribution")
    ax[2].set_xlabel("Error")
    ax[2].set_ylabel("Frequency")

    for axis in ax:
        axis.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)

def evaluate_var_statistics(model_name, true_data, predicted_data, variables):

    st.subheader(f"Model: {model_name}")
    if isinstance(true_data, np.ndarray):
        true_data = pd.DataFrame(true_data, columns=variables)
    if isinstance(predicted_data, np.ndarray):
        predicted_data = pd.DataFrame(predicted_data, columns=variables)

    min_length = min(len(true_data), len(predicted_data))
    true_data = true_data.iloc[:min_length].reset_index(drop=True)
    predicted_data = predicted_data.iloc[:min_length].reset_index(drop=True)

    # Initialize subplots for each variable
    n_variables = len(variables)
    fig, axes = plt.subplots(n_variables, 1, figsize=(12, 6 * n_variables))

    # Compute and plot statistics for each variable
    for i, variable in enumerate(variables):
        true_values = true_data[variable]
        predicted_values = predicted_data[variable]

        # Calculate residuals
        residuals = true_values - predicted_values

        # RMSE, MAE, MBE
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        mae = np.mean(np.abs(residuals))
        mbe = np.mean(residuals)

        # Display metrics
        st.subheader(f"{model_name} - {variable}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MBE: {mbe:.2f}")

        # Plot residuals
        ax = axes[i] if n_variables > 1 else axes
        ax.plot(true_values.index, residuals, label=f"Residuals ({variable})", color="purple")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{model_name} Residuals for {variable}")
        ax.set_ylabel("Residuals")
        ax.legend()

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)


def delta_page():

    st.subheader("Delta Airlines Overview")
    st.write("""
        **What:** This page analyzes delays in Delta Airlines caused by weather and holidays.
        **Why:** Delta can improve operations by understanding these patterns.
        **How:** Using interactive visualizations and predictive models.
    """)
    st.title("Delta Airlines")
    st.write("""
        This page explores how Delta Airlines is impacted by weather conditions and federal holidays.

        ## Sections:
        - **Visualizations**: Analyze how different factors affect delays and cancellations.
        - **Prediction Model**: Predict arrival delays based on weather and holiday data.
        - **Statistical Analysis**: Explore correlations between variables and delays.
    """)

    # Load Delta data
    non_cancelled_data, cancelled_data = load_airline_data('Delta')

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=non_cancelled_data['FL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=non_cancelled_data['FL_DATE'].max())

    # Filter by date range
    filtered_non_cancelled = non_cancelled_data[(non_cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                                (non_cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]
    filtered_cancelled = cancelled_data[(cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                        (cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]

    # Display filtered data
    st.subheader("Filtered Non-Cancelled Flights")
    st.write(filtered_non_cancelled)

    st.subheader("Filtered Cancelled Flights")
    st.write(filtered_cancelled)

    # Convert categorical severity columns to numeric values for correlation and visualization
    severity_map = {'Light': 1, 'Moderate': 2, 'Heavy': 3}
    filtered_non_cancelled['Severity_origin_numeric'] = filtered_non_cancelled['Severity_origin'].map(severity_map)
    filtered_non_cancelled['Severity_dest_numeric'] = filtered_non_cancelled['Severity_dest'].map(severity_map)

    # Visualizations Section
    st.header("Visualizations")

    # Correlation Matrix (Heatmap)
    st.subheader("Correlation Matrix")
    plot_correlation_matrix(filtered_non_cancelled)

    # Severity vs. Delay Visualizations
    st.subheader("Severity vs. Delay")
    plot_severity_origin_dep_delay(filtered_non_cancelled)
    plot_severity_dest_arr_delay(filtered_non_cancelled)

    # Holiday Impact
    st.subheader("Holiday Impact on Delays")
    plot_delay_by_holiday(filtered_non_cancelled)

    # Delay Distribution
    st.subheader("Delay Distribution")
    plot_delay_distribution(filtered_non_cancelled)

    # Average Delay Over Time
    st.subheader("Average Delay Over Time")
    plot_avg_delay_over_time(filtered_non_cancelled)

    # Weather Impact by Airports
    st.subheader("Weather Impact by Airports (Select the Airport Pair on the Left Side Tool Bar)")
    plot_weather_impact_by_airports(filtered_non_cancelled)

    # Prediction Models Section
    st.header("Prediction Models")

    # Sidebar settings for predictions
    st.sidebar.header("Prediction Settings")
    forecast_type = st.sidebar.selectbox("Select Forecast Type", ['Mean', 'Weather Features', 'Holiday'])
    p_value = st.sidebar.slider("AR(p) Lag Value", min_value=1, max_value=10, value=1)

    # Prediction Section
    st.subheader("Predict Arrival Delays")
    st.write("""
        Select prediction parameters in the sidebar to forecast arrival delays.
    """)

    # Sidebar inputs for lag and forecast steps
    lag_range = st.sidebar.slider("Select AR Lag Range for ACF/PACF", min_value=1, max_value=10, value=1)
    forecast_steps = st.sidebar.slider("Forecast Steps", min_value=1, max_value=10, value=1)

    # ACF and PACF Analysis
    train_data = filtered_non_cancelled['ARR_DELAY'][:int(len(filtered_non_cancelled) * 0.8)]  # 80% train split
    acf_values = acf(train_data, nlags=lag_range)
    pacf_values = pacf(train_data, nlags=lag_range)

    st.subheader("Autocorrelation Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(range(len(acf_values)), acf_values)
    ax[0].set_title("ACF")
    ax[1].bar(range(len(pacf_values)), pacf_values)
    ax[1].set_title("PACF")
    st.pyplot(fig)

""    # Fit AR(p) and Forecast
    forecast = fit_ar_model(train_data.values, lag_range, forecast_steps)

    # Evaluation and Plot
    test_data = filtered_non_cancelled['ARR_DELAY'][int(len(filtered_non_cancelled) * 0.8):]
    rmse = np.sqrt(mean_squared_error(test_data[:forecast_steps], forecast))
    st.write(f"RMSE: {rmse:.2f}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 12)) ""

    # Full plot (top)
    ax[0].plot(train_data.index, train_data, label="Training Data", color="blue")
    ax[0].plot(test_data.index[:forecast_steps], test_data[:forecast_steps], label="Actual Test Data", color="orange")
    ax[0].plot(test_data.index[:forecast_steps], forecast, label="Forecast", linestyle="--", color="green")
    ax[0].set_title("AR(p) Forecast vs Actual")
    ax[0].set_ylabel("Arrival Delay")
    ax[0].legend()

    # Zoomed-in plot (bottom)
    ax[1].plot(test_data.index[-forecast_steps:], test_data[-forecast_steps:], label="Actual Test Data", color="orange")
    ax[1].plot(test_data.index[-forecast_steps:], forecast, label="Forecast", linestyle="--", color="green")
    ax[1].set_title("Zoomed-in Forecast vs Actual")

    # Adjust the x-axis limits for the zoomed-in plot to focus on the last portion of data
    ax[1].set_xlim(test_data.index[-forecast_steps], test_data.index[-1])

    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Arrival Delay")
    ax[1].legend()

    for axis in ax:
        axis.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)

    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    # Allow the user to choose the lag (p_value) dynamically
    p_value = st.slider("Select the lag (p) for the AR model:", min_value=1, max_value=10, value=1)

    forecast_steps = st.slider("Select the number of forecast steps:", min_value=1, max_value=10, value=1)

    forecast, predictions, train_data, test_data = forecast_ar_model_with_mean(filtered_non_cancelled, p_value=p_value,
                                                                               forecast_steps=forecast_steps)
    plot_ar_model_with_mean(train_data, test_data, predictions, forecast, p_value=p_value)

    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")
    # User input for the lag value (p) and forecast steps
    p_value_var = st.slider("Select Lag (p) value for VAR(p)", 1, 1, 1)  # Default p = 2


    forecast, results, train_data, test_data = forecast_var_model(filtered_non_cancelled, p_value_var)
    plot_var_results(train_data, test_data, forecast, p_value_var)

    st.header("Evaluation and Comparison of Models")

    # AR(p) Model Evaluation
    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    # Run AR(p) Model
    forecast_ar, predictions_ar, train_data_ar, test_data_ar = forecast_ar_model_with_mean(
        filtered_non_cancelled, p_value=p_value, forecast_steps=forecast_steps
    )

    # Evaluate AR(p) Model
    evaluate_and_plot_statistics(
        model_name="AR(p) Model (Aggregated Mean Arrival Delays)",
        true_values=test_data_ar[:forecast_steps],  # Align with forecast_steps
        predicted_values=predictions_ar,
    )

    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")

    forecast_var, predictions_var, train_data_var, test_data_var = forecast_var_model(
        filtered_non_cancelled, p_value_var
    )

    var_variables = ["ARR_DELAY", "DEP_DELAY", "AIR_TIME"]

    evaluate_var_statistics(
        model_name=f"VAR({p_value_var}) Model",
        true_data=test_data_var,
        predicted_data=forecast_var,
        variables=var_variables
    )

    st.header("Summary of Model Performance")
    st.write("The evaluation statistics provide insights into how well each model predicts ARR_DELAY. "
             "The RMSE, MAE, and MBE metrics, along with residual plots, help assess the quality of the forecasts.")

    st.header("Take Homes for Consumer and Airline:")
    st.write("Delta Airlines shows a strong correlation between severe weather conditions and flight delays, "
             "particularly at both origin and destination airports. Federal holidays further amplify these delays due "
             "to increased passenger volume. By leveraging weather and holiday data, Delta can enhance its operational "
             "planning and reduce the impact of delays on its customers."
             ""
             "Future Applications for this app could include expanding the modeling to include real-time flight data"
             "sourced from another company such as FlightRadar24, this data could be passed to the model and give consumers"
             "and airlines real-time information on what flights might be delayed."
             
             "Furthermore, with more advanced machine learning algorithms the potential for more accurate models will drastically enhance the commercialization of this app"             )


# American Airlines page
def american_page():

    st.subheader("American Airlines Overview")
    st.write("""
        **What:** Analyzes how weather and holidays impact American Airlines' flights.
        **Why:** Enables operational efficiency improvements.
        **How:** Offers data-driven insights through visualization and modeling.
    """)
    st.title("American Airlines")
    st.write("""
    This page explores how American Airlines is impacted by weather conditions and federal holidays.

        ## Visualizations:
        - **Weather Impact**: Analyze how different weather features (e.g., precipitation, severity) affect delays and cancellations for Delta.
        - **Holiday Impact**: Compare delay patterns on federal holidays vs. regular days.

        ## Prediction Model:
        - Predict delays or cancellations based on selected weather conditions or holiday status.

        ## Statistical Analysis:
        - Understand the correlation between weather variables and flight delays/cancellations.
        """
             )

    # Load American Airlines data
    non_cancelled_data, cancelled_data = load_airline_data('American')

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=non_cancelled_data['FL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=non_cancelled_data['FL_DATE'].max())

    # Filter by date range
    filtered_non_cancelled = non_cancelled_data[(non_cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                                (non_cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]
    filtered_cancelled = cancelled_data[(cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                        (cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]

    # Apply the severity mapping to numeric values
    severity_map = {'Light': 1, 'Moderate': 2, 'Heavy': 3}
    filtered_non_cancelled['Severity_origin_numeric'] = filtered_non_cancelled['Severity_origin'].map(severity_map)
    filtered_non_cancelled['Severity_dest_numeric'] = filtered_non_cancelled['Severity_dest'].map(severity_map)

    # Visualizations
    st.subheader("Weather Impact Visualizations")
    plot_correlation_matrix(filtered_non_cancelled)
    plot_severity_origin_dep_delay(filtered_non_cancelled)
    plot_severity_dest_arr_delay(filtered_non_cancelled)
    plot_delay_by_holiday(filtered_non_cancelled)
    plot_delay_distribution(filtered_non_cancelled)
    plot_avg_delay_over_time(filtered_non_cancelled)
    plot_weather_impact_by_airports(filtered_non_cancelled)

    # Prediction Models Section
    st.header("Prediction Models")

    # Sidebar settings for predictions
    st.sidebar.header("Prediction Settings")
    forecast_type = st.sidebar.selectbox("Select Forecast Type", ['Mean', 'Weather Features', 'Holiday'])
    p_value = st.sidebar.slider("AR(p) Lag Value", min_value=1, max_value=10, value=4)

    # Prediction Section
    st.subheader("Predict Arrival Delays")
    st.write("""
            Select prediction parameters in the sidebar to forecast arrival delays.
        """)

    # Sidebar inputs for lag and forecast steps
    lag_range = st.sidebar.slider("Select AR Lag Range for ACF/PACF", min_value=1, max_value=10, value=4)
    forecast_steps = st.sidebar.slider("Forecast Steps", min_value=1, max_value=10, value=4)

    # ACF and PACF Analysis
    train_data = filtered_non_cancelled['ARR_DELAY'][:int(len(filtered_non_cancelled) * 0.8)]  # 80% train split
    acf_values = acf(train_data, nlags=lag_range)
    pacf_values = pacf(train_data, nlags=lag_range)

    st.subheader("Autocorrelation Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(range(len(acf_values)), acf_values)
    ax[0].set_title("ACF")
    ax[1].bar(range(len(pacf_values)), pacf_values)
    ax[1].set_title("PACF")
    st.pyplot(fig)

    # Fit AR(p) and Forecast
    forecast = fit_ar_model(train_data.values, lag_range, forecast_steps)

    # Evaluation and Plot
    test_data = filtered_non_cancelled['ARR_DELAY'][int(len(filtered_non_cancelled) * 0.8):]
    rmse = np.sqrt(mean_squared_error(test_data[:forecast_steps], forecast))
    st.write(f"RMSE: {rmse:.2f}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # Full plot (top)
    ax[0].plot(train_data.index, train_data, label="Training Data", color="blue")
    ax[0].plot(test_data.index[:forecast_steps], test_data[:forecast_steps], label="Actual Test Data", color="orange")
    ax[0].plot(test_data.index[:forecast_steps], forecast, label="Forecast", linestyle="--", color="green")
    ax[0].set_title("AR(p) Forecast vs Actual")
    ax[0].set_ylabel("Arrival Delay")
    ax[0].legend()

    ax[1].plot(test_data.index[-forecast_steps:], test_data[-forecast_steps:], label="Actual Test Data", color="orange")
    ax[1].plot(test_data.index[-forecast_steps:], forecast, label="Forecast", linestyle="--", color="green")
    ax[1].set_title("Zoomed-in Forecast vs Actual")

    ax[1].set_xlim(test_data.index[-forecast_steps], test_data.index[-1])

    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Arrival Delay")
    ax[1].legend()

    for axis in ax:
        axis.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)

    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    # Allow the user to choose the lag (p_value) dynamically
    p_value = st.slider("Select the lag (p) for the AR model:", min_value=1, max_value=30, value=20)

    # Allow user to adjust the number of forecast steps
    forecast_steps = st.slider("Select the number of forecast steps:", min_value=1, max_value=10, value=5)

    forecast, predictions, train_data, test_data = forecast_ar_model_with_mean(filtered_non_cancelled, p_value=p_value,
                                                                               forecast_steps=forecast_steps)
    plot_ar_model_with_mean(train_data, test_data, predictions, forecast, p_value=p_value)

    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")
    # User input for the lag value (p) and forecast steps
    p_value_var = st.slider("Select Lag (p) value for VAR(p)", 1, 10, 2)  # Default p = 2


    forecast, results, train_data, test_data = forecast_var_model(filtered_non_cancelled, p_value_var)
    plot_var_results(train_data, test_data, forecast, p_value_var)

    st.header("Evaluation and Comparison of Models")

    # AR(p) Model Evaluation
    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    forecast_ar, predictions_ar, train_data_ar, test_data_ar = forecast_ar_model_with_mean(
        filtered_non_cancelled, p_value=p_value, forecast_steps=forecast_steps
    )

    evaluate_and_plot_statistics(
        model_name="AR(p) Model (Aggregated Mean Arrival Delays)",
        true_values=test_data_ar[:forecast_steps],  # Align with forecast_steps
        predicted_values=predictions_ar,
    )

    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")

    forecast_var, predictions_var, train_data_var, test_data_var = forecast_var_model(
        filtered_non_cancelled, p_value_var
    )

    var_variables = ["ARR_DELAY", "DEP_DELAY", "AIR_TIME"]

    evaluate_var_statistics(
        model_name=f"VAR({p_value_var}) Model",
        true_data=test_data_var,
        predicted_data=forecast_var,
        variables=var_variables
    )

    st.header("Summary of Model Performance")
    st.write("The evaluation statistics provide insights into how well each model predicts ARR_DELAY. "
             "The RMSE, MAE, and MBE metrics, along with residual plots, help assess the quality of the forecasts.")

    st.header("Take Homes for Consumer and Airline:")
    st.write("American Airlines experiences significant delays during heavy precipitation and on federal holidays. The data reveals that proactive measures around peak travel periods and adverse weather can help mitigate delays. Insights from the predictive models enable American Airlines to anticipate delays more accurately and enhance customer satisfaction."
             
             "Future Applications for this app could include expanding the modeling to include real-time flight data"
             "sourced from another company such as FlightRadar24, this data could be passed to the model and give consumers"
             "and airlines real-time information on what flights might be delayed."

             "Furthermore, with more advanced machine learning algorithms the potential for more accurate models will drastically enhance the commercialization of this app")



# United Airlines page
def united_page():

    st.subheader("United Airlines Overview")
    st.write("""
        **What:** Examines the impact of weather and holidays on United Airlines' delays.
        **Why:** Helps United mitigate delay risks and enhance customer satisfaction.
        **How:** Uses data visualization and statistical models.
    """)
    st.title("United Airlines")
    st.write("""This page explores how United Airlines is impacted by weather conditions and federal holidays.

        ## Visualizations:
        - **Weather Impact**: Analyze how different weather features (e.g., precipitation, severity) affect delays and cancellations for United Airlines.
        - **Holiday Impact**: Compare delay patterns on federal holidays vs. regular days.

        ## Prediction Model:
        - Predict delays or cancellations based on selected weather conditions or holiday status.

        ## Statistical Analysis:
        - Understand the correlation between weather variables and flight delays/cancellations.
        """)

    # Load United Airlines data
    non_cancelled_data, cancelled_data = load_airline_data('United')

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=non_cancelled_data['FL_DATE'].min())
    end_date = st.sidebar.date_input("End Date", value=non_cancelled_data['FL_DATE'].max())

    # Filter by date range
    filtered_non_cancelled = non_cancelled_data[(non_cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                                (non_cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]
    filtered_cancelled = cancelled_data[(cancelled_data['FL_DATE'] >= pd.to_datetime(start_date)) &
                                        (cancelled_data['FL_DATE'] <= pd.to_datetime(end_date))]

    # Apply the severity mapping to numeric values
    severity_map = {'Light': 1, 'Moderate': 2, 'Heavy': 3}
    filtered_non_cancelled['Severity_origin_numeric'] = filtered_non_cancelled['Severity_origin'].map(severity_map)
    filtered_non_cancelled['Severity_dest_numeric'] = filtered_non_cancelled['Severity_dest'].map(severity_map)

    # Visualizations
    st.subheader("Weather Impact Visualizations")
    plot_correlation_matrix(filtered_non_cancelled)
    plot_severity_origin_dep_delay(filtered_non_cancelled)
    plot_severity_dest_arr_delay(filtered_non_cancelled)
    plot_delay_by_holiday(filtered_non_cancelled)
    plot_delay_distribution(filtered_non_cancelled)
    plot_avg_delay_over_time(filtered_non_cancelled)
    plot_weather_impact_by_airports(filtered_non_cancelled)

    # Prediction Models Section
    st.header("Prediction Models")

    # Sidebar settings for predictions
    st.sidebar.header("Prediction Settings")
    forecast_type = st.sidebar.selectbox("Select Forecast Type", ['Mean', 'Weather Features', 'Holiday'])
    p_value = st.sidebar.slider("AR(p) Lag Value", min_value=1, max_value=10, value=4)

    # Prediction Section
    st.subheader("Predict Arrival Delays")
    st.write("""
            Select prediction parameters in the sidebar to forecast arrival delays.
        """)

    lag_range = st.sidebar.slider("Select AR Lag Range for ACF/PACF", min_value=1, max_value=10, value=4)
    forecast_steps = st.sidebar.slider("Forecast Steps", min_value=1, max_value=10, value=4)

    train_data = filtered_non_cancelled['ARR_DELAY'][:int(len(filtered_non_cancelled) * 0.8)]  # 80% train split
    acf_values = acf(train_data, nlags=lag_range)
    pacf_values = pacf(train_data, nlags=lag_range)

    st.subheader("Autocorrelation Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(range(len(acf_values)), acf_values)
    ax[0].set_title("ACF")
    ax[1].bar(range(len(pacf_values)), pacf_values)
    ax[1].set_title("PACF")
    st.pyplot(fig)

    forecast = fit_ar_model(train_data.values, lag_range, forecast_steps)

    test_data = filtered_non_cancelled['ARR_DELAY'][int(len(filtered_non_cancelled) * 0.8):]
    rmse = np.sqrt(mean_squared_error(test_data[:forecast_steps], forecast))
    st.write(f"RMSE: {rmse:.2f}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # Full plot (top)
    ax[0].plot(train_data.index, train_data, label="Training Data", color="blue")
    ax[0].plot(test_data.index[:forecast_steps], test_data[:forecast_steps], label="Actual Test Data", color="orange")
    ax[0].plot(test_data.index[:forecast_steps], forecast, label="Forecast", linestyle="--", color="green")
    ax[0].set_title("AR(p) Forecast vs Actual")
    ax[0].set_ylabel("Arrival Delay")
    ax[0].legend()

    # Zoomed-in plot (bottom)
    ax[1].plot(test_data.index[-forecast_steps:], test_data[-forecast_steps:], label="Actual Test Data", color="orange")
    ax[1].plot(test_data.index[-forecast_steps:], forecast, label="Forecast", linestyle="--", color="green")
    ax[1].set_title("Zoomed-in Forecast vs Actual")

    # Adjust the x-axis limits for the zoomed-in plot to focus on the last portion of data
    ax[1].set_xlim(test_data.index[-forecast_steps], test_data.index[-1])

    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Arrival Delay")
    ax[1].legend()

    for axis in ax:
        axis.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)

    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    # Allow the user to choose the lag (p_value) dynamically
    p_value = st.slider("Select the lag (p) for the AR model:", min_value=1, max_value=30, value=20)

    # Allow user to adjust the number of forecast steps
    forecast_steps = st.slider("Select the number of forecast steps:", min_value=1, max_value=10, value=5)

    forecast, predictions, train_data, test_data = forecast_ar_model_with_mean(filtered_non_cancelled, p_value=p_value,
                                                                               forecast_steps=forecast_steps)
    plot_ar_model_with_mean(train_data, test_data, predictions, forecast, p_value=p_value)

    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")
    # User input for the lag value (p) and forecast steps
    p_value_var = st.slider("Select Lag (p) value for VAR(p)", 1, 10, 2)  # Default p = 2


    forecast, results, train_data, test_data = forecast_var_model(filtered_non_cancelled, p_value_var)
    plot_var_results(train_data, test_data, forecast, p_value_var)

    st.header("Evaluation and Comparison of Models")

    # AR(p) Model Evaluation
    st.header("AR(p) Model Forecast on Aggregated Mean Arrival Delays")

    # Run AR(p) Model
    forecast_ar, predictions_ar, train_data_ar, test_data_ar = forecast_ar_model_with_mean(
        filtered_non_cancelled, p_value=p_value, forecast_steps=forecast_steps
    )

    evaluate_and_plot_statistics(
        model_name="AR(p) Model (Aggregated Mean Arrival Delays)",
        true_values=test_data_ar[:forecast_steps],  # Align with forecast_steps
        predicted_values=predictions_ar,
    )

    # VAR(p) Model Analysis
    st.header("VAR(p) Model Analysis")
    st.subheader("Forecasting ARR_DELAY Using Average DEP_DELAY and Average AIR_TIME")

    forecast_var, predictions_var, train_data_var, test_data_var = forecast_var_model(
        filtered_non_cancelled, p_value_var
    )


    var_variables = ["ARR_DELAY", "DEP_DELAY", "AIR_TIME"]


    evaluate_var_statistics(
        model_name=f"VAR({p_value_var}) Model",
        true_data=test_data_var,
        predicted_data=forecast_var,
        variables=var_variables
    )

    # Final Layout and Analysis Summary
    st.header("Summary of Model Performance")
    st.write("The evaluation statistics provide insights into how well each model predicts ARR_DELAY. "
             "The RMSE, MAE, and MBE metrics, along with residual plots, help assess the quality of the forecasts.")

    st.header("Take Homes for Consumer and Airline:")
    st.write(
        "United Airlines exhibits a clear relationship between severe weather, such as heavy precipitation, and extended delays, especially during federal holidays. The analysis highlights the importance of incorporating weather forecasts and holiday schedules into operational strategies. By doing so, United Airlines can improve on-time performance and better manage customer expectations during high-risk periods."
        "United consumers can also use this app to better understand how thier flights may be affect prior ot arriving at the airport"

        "Future Applications for this app could include expanding the modeling to include real-time flight data"
        "sourced from another company such as FlightRadar24, this data could be passed to the model and give consumers"
        "and airlines real-time information on what flights might be delayed."

        "Furthermore, with more advanced machine learning algorithms the potential for more accurate models will drastically enhance the commercialization of this app")


# Page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Delta Airlines", "American Airlines", "United Airlines"])

    if page == "Home":
        homepage()
    elif page == "Delta Airlines":
        delta_page()
    elif page == "American Airlines":
        american_page()
    elif page == "United Airlines":
        united_page()


if __name__ == "__main__":
    main()
