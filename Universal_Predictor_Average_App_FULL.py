import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os

# ==== 1. Generate data and save to file ====

def generate_data():
    years = np.arange(2020, 2040)
    months = np.arange(1, 13)
    date_combinations = [(year, month) for year in years for month in months]
    values = np.random.uniform(1, 1000000, len(date_combinations))
    df = pd.DataFrame(date_combinations, columns=["year", "month"])
    df["value"] = values
    df["average_value"] = df.groupby("year")["value"].transform("mean")
    return df

df = generate_data()
file_path = "universal_data_with_average.xlsx"
df.to_excel(file_path, index=False)

# ==== 2. Load data and train model ====

def load_data(file_path):
    return pd.read_excel(file_path)

df = load_data(file_path)

target_column = "value"
feature_columns = ["year", "month"]

X = df[feature_columns]
y = df[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "rf_scaler.pkl")

# ==== 3. Prediction function ====

def predict():
    try:
        year = int(year_entry.get())
        month = int(month_entry.get())
        period = int(period_combobox.get())

        predictions = []
        for i in range(period):
            new_month = (month + i - 1) % 12 + 1
            new_year = year + (month + i - 1) // 12
            X_input = np.array([[new_year, new_month]])
            X_input_scaled = scaler.transform(X_input)
            prediction = model.predict(X_input_scaled)[0]
            predictions.append((f"{new_year}-{new_month:02d}", prediction))

        result_label.config(text="📈 Forecast for the upcoming months:")

        for widget in graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 6))
        months = [x[0] for x in predictions]
        values = [x[1] for x in predictions]

        ax.bar(months, values, color="orange")
        ax.set_ylabel("Index (PPI)")
        ax.set_title(f"Forecast for the next {period} months")
        plt.xticks(rotation=45, ha="right")

        average_prediction = np.mean(values)
        ax.axhline(y=average_prediction, color='r', linestyle='--', label=f'Average prediction: {average_prediction:.2f}')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # === Save to log file ===
        log_file = "prediction_log.xlsx"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_df = pd.DataFrame(predictions, columns=["Period", "Predicted Value"])
        log_df["DateTime"] = timestamp
        log_df["Input_Year"] = year
        log_df["Input_Month"] = month
        log_df["Forecast_Months"] = period
        log_df["Average_Prediction"] = average_prediction
        log_df = log_df[["DateTime", "Input_Year", "Input_Month", "Forecast_Months", "Period", "Predicted Value", "Average_Prediction"]]

        if os.path.exists(log_file):
            existing_df = pd.read_excel(log_file)
            combined_df = pd.concat([existing_df, log_df], ignore_index=True)
        else:
            combined_df = log_df

        combined_df.to_excel(log_file, index=False)
        print("📁 Saved to prediction_log.xlsx")

    except Exception as e:
        result_label.config(text=f"Error: {e}")

# ==== 4. Historical data analysis function ====

def analyze_data():
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.sort_values(by=["year", "month"])
    x_vals = np.arange(len(df_sorted))
    y_vals = df_sorted["value"].values

    ax.plot(x_vals, y_vals, label="Historical values", color="blue", marker='o')

    linreg = LinearRegression()
    linreg.fit(x_vals.reshape(-1, 1), y_vals)
    trend = linreg.predict(x_vals.reshape(-1, 1))
    ax.plot(x_vals, trend, label="Trend line", linestyle='--', color='red')

    ax.set_title("Historical Data Analysis")
    ax.set_ylabel("Value")
    ax.set_xlabel("Months (sorted)")
    ax.legend()
    plt.xticks([])

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# ==== 5. Create GUI ====

root = tk.Tk()
root.title("Forecast and Value Analysis 2.5 by Hexagon Lab")
root.minsize(900, 650)

# Set the window icon (replace 'app_icon.ico' with the path to your .ico file)
root.iconbitmap('business-graphic.ico')  # Make sure this file exists in the same directory as your script

# Center the window
window_width = 1024
window_height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int((screen_width - window_width) / 2)
center_y = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

# Input fields
tk.Label(root, text="Year:").grid(row=0, column=0)
year_entry = tk.Entry(root)
year_entry.grid(row=0, column=1)

tk.Label(root, text="Month (1-12):").grid(row=1, column=0)
month_entry = tk.Entry(root)
month_entry.grid(row=1, column=1)

tk.Label(root, text="Period (months):").grid(row=2, column=0)
period_combobox = ttk.Combobox(root, values=[4, 5, 6, 7, 8, 9, 10, 11, 12], state="readonly")
period_combobox.grid(row=2, column=1)
period_combobox.set(4)

# Buttons
ttk.Button(root, text="Predict", command=predict).grid(row=3, column=0, columnspan=2, pady=10)
ttk.Button(root, text="Analyze Data", command=analyze_data).grid(row=4, column=0, columnspan=2, pady=5)

# Result display
result_label = tk.Label(root, text="Enter data and press the button")
result_label.grid(row=5, column=0, columnspan=2)

# Graph frame
graph_frame = tk.Frame(root)
graph_frame.grid(row=6, column=0, columnspan=2)

root.mainloop()
