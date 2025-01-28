import streamlit as st
import json
import requests

st.title('Credit scoring of Bati Bank')
# Streamlit app

# Streamlit app


st.title("User Input Form")

st.header("Fill in the details below:")
# Create a form
with st.form("user_input_form"):
    # Input fields
    transaction_id = st.number_input("Transaction ID", min_value=1, step=1)
    customer_id = st.number_input("Customer ID", min_value=1, step=1)
    provider_id = st.number_input("Provider ID", min_value=1, step=1)
    product_id = st.number_input("Product ID", min_value=1, step=1)
    product_category = st.selectbox("Product Category", options=[
                                    "Category1", "Category2", "Category3"])  # Adjust options as needed
    channel_id = st.number_input("Channel ID", min_value=1, step=1)
    amount = st.number_input("Amount", min_value=0.0, step=0.1)
    value = st.number_input("Value", min_value=0.0, step=0.1)
    std_dev_of_transaction_amounts = st.number_input(
        "Standard Deviation of Transaction Amounts", min_value=0.0, step=0.1)
    transaction_count = st.number_input(
        "Transaction Count", min_value=0, step=1)
    avg_transaction_amount = st.number_input(
        "Average Transaction Amount", min_value=0.0, step=0.1)
    total_transaction_amount = st.number_input(
        "Total Transaction Amount", min_value=0.0, step=0.1)
    transaction_year = st.number_input(
        "Transaction Year", min_value=1900, step=1)
    transaction_month = st.number_input(
        "Transaction Month", min_value=1, max_value=12, step=1)
    transaction_day = st.number_input(
        "Transaction Day", min_value=1, max_value=31, step=1)
    transaction_hour = st.number_input(
        "Transaction Hour", min_value=0, max_value=23, step=1)
    pricing_strategy = st.selectbox("Pricing Strategy", options=[
                                    "Strategy1", "Strategy2", "Strategy3"])  # Adjust options as needed

    # Submit button
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Form submitted successfully!")
    user_input = {
        {
            "TransactionId": transaction_id,
            "CustomerId": customer_id,
            "ProviderId": provider_id,
            "ProductId": product_id,
            "ProductCategory": product_category,
            "ChannelId": channel_id,
            "Amount": amount,
            "Value": value,
            "Standard-Deviation-of-Transaction-Amounts": std_dev_of_transaction_amounts,
            "Transaction-Count": transaction_count,
            "Average-Transaction-Amount": avg_transaction_amount,
            "Total-Transaction-Amount": total_transaction_amount,
            "Transaction-Year": transaction_year,
            "Transaction-Month": transaction_month,
            "Transaction-Day": transaction_day,
            "Transaction-Hour": transaction_hour,
            "PricingStrategy": pricing_strategy,

        }

    }
    inputs = st.json(user_input)  # Display the input in JSON format
    st.session_state.inputs = inputs

if st.button('Classify'):
    inputs = st.session_state.inputs
    res = requests.post(
        url='http://127.0.0.1:8000/predict', data=json.dumps(inputs))
    st.subheader(
        f'Response from API call (Predicted result) is = {res.text}')
