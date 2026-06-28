import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Sales Trend Dashboard", layout="wide")


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        compression = "zip" if str(uploaded_file).lower().endswith(".zip") else None
        data = pd.read_csv(uploaded_file, compression=compression)
    else:
        possible_files = [
            "cleaned_sales_data.zip",
            "../cleaned_sales_data.zip",
            "Online Retail.csv",
            "../Online Retail.csv",
        ]
        file_path = next((path for path in possible_files if os.path.exists(path)), None)
        if file_path is None:
            return None
        compression = "zip" if file_path.lower().endswith(".zip") else None
        data = pd.read_csv(file_path, compression=compression)

    data.columns = data.columns.str.strip()

    if "InvoiceDate" not in data.columns:
        return "missing_date"

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    if "Sales" not in data.columns:
        if {"Quantity", "UnitPrice"}.issubset(data.columns):
            data["Sales"] = data["Quantity"] * data["UnitPrice"]
        else:
            return "missing_sales"

    data["Year"] = data["InvoiceDate"].dt.year
    data["Month"] = data["InvoiceDate"].dt.month_name()
    data["Month_Number"] = data["InvoiceDate"].dt.month
    data["Weekday"] = data["InvoiceDate"].dt.day_name()

    return data


st.title("Sales Trend Visualization Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload cleaned_sales_data.zip", type=["zip"])
df = load_data(uploaded_file)

if df is None:
    st.warning("No data found. Upload `cleaned_sales_data.zip` from the sidebar or keep it in the same folder as this dashboard.")
    st.stop()

if isinstance(df, str):
    if df == "missing_date":
        st.error("Your dataset must contain an `InvoiceDate` column.")
    elif df == "missing_sales":
        st.error("Your dataset must contain `Sales`, or both `Quantity` and `UnitPrice` columns.")
    st.stop()

required_columns = ["InvoiceNo", "Country", "Description", "Sales", "InvoiceDate"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

countries = sorted(df["Country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=["United Kingdom"] if "United Kingdom" in countries else countries[:3],
)

top_n = st.sidebar.slider("Top Products", min_value=5, max_value=30, value=10)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = df[
    (df["InvoiceDate"].dt.date >= start_date)
    & (df["InvoiceDate"].dt.date <= end_date)
    & (df["Country"].isin(selected_countries))
]

if filtered_df.empty:
    st.warning("No records found for the selected filters. Try selecting more countries or a wider date range.")
    st.stop()

total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["InvoiceNo"].nunique()
total_customers = filtered_df["CustomerID"].nunique() if "CustomerID" in filtered_df.columns else 0
avg_order_value = total_sales / total_orders if total_orders else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Average Order Value", f"{avg_order_value:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sales Trend", "Products", "Countries", "Customer/Orders", "Forecast"]
)

with tab1:
    monthly_sales = (
        filtered_df.resample("ME", on="InvoiceDate")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="InvoiceDate",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
    )
    st.plotly_chart(fig, use_container_width=True)

    weekday_sales = (
        filtered_df.groupby("Weekday")["Sales"]
        .sum()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .reset_index()
    )

    fig = px.bar(
        weekday_sales,
        x="Weekday",
        y="Sales",
        title="Sales by Weekday",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    top_products = (
        filtered_df.groupby("Description")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Description",
        orientation="h",
        title=f"Top {top_n} Products by Sales",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    country_sales = (
        filtered_df.groupby("Country")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        country_sales,
        x="Country",
        y="Sales",
        title="Country-wise Sales",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    monthly_orders = (
        filtered_df.resample("ME", on="InvoiceDate")["InvoiceNo"]
        .nunique()
        .reset_index()
    )

    fig = px.line(
        monthly_orders,
        x="InvoiceDate",
        y="InvoiceNo",
        markers=True,
        title="Monthly Order Trend",
    )
    st.plotly_chart(fig, use_container_width=True)

    if "CustomerID" in filtered_df.columns:
        top_customers = (
            filtered_df.groupby("CustomerID")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            top_customers,
            x="CustomerID",
            y="Sales",
            title="Top 10 Customers by Sales",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    forecast_data = (
        filtered_df.resample("ME", on="InvoiceDate")["Sales"]
        .sum()
        .reset_index()
    )

    if len(forecast_data) < 3:
        st.warning("At least 3 months of data are needed for forecasting.")
    else:
        forecast_data["Time_Index"] = np.arange(len(forecast_data))
        model = LinearRegression()
        model.fit(forecast_data[["Time_Index"]], forecast_data["Sales"])

        future_index = np.arange(len(forecast_data), len(forecast_data) + 6)
        future_dates = pd.date_range(
            start=forecast_data["InvoiceDate"].max() + pd.DateOffset(months=1),
            periods=6,
            freq="ME",
        )
        future_sales = model.predict(pd.DataFrame({"Time_Index": future_index}))

        forecast_df = pd.DataFrame(
            {
                "InvoiceDate": future_dates,
                "Sales": future_sales,
                "Type": "Predicted",
            }
        )
        actual_df = forecast_data[["InvoiceDate", "Sales"]].copy()
        actual_df["Type"] = "Actual"

        combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)

        fig = px.line(
            combined_df,
            x="InvoiceDate",
            y="Sales",
            color="Type",
            markers=True,
            title="Actual vs Predicted Sales",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast_df, use_container_width=True)

st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Data",
    data=csv_data,
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
