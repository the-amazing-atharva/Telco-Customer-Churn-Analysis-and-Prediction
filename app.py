import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(layout="wide", page_title="Telco Churn Analysis")

st.title("Telco Customer Churn Analysis Dashboard by Atharva Salitri")
st.write("This interactive dashboard presents key visualizations and insights from the Telco Customer Churn analysis project.")

# --- Data Loading and Cleaning (as performed in the notebook) ---


@st.cache_data(show_spinner=False)  # Cache data loading to improve performance
def load_and_preprocess_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    # Preprocessing for ML models (for correlation heatmap and future ML sections)
    df_processed = df.copy()

    binary_cols = ['Partner', 'Dependents',
                   'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
    df_processed['gender'] = df_processed['gender'].map(
        {'Female': 1, 'Male': 0})

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in service_cols:
        df_processed[col] = df_processed[col].replace(
            {'No internet service': 'No', 'No phone service': 'No'})
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    multi_class_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df_processed = pd.get_dummies(
        df_processed, columns=multi_class_cols, drop_first=True)

    numerical_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df_processed[numerical_cols_to_scale] = scaler.fit_transform(
        df_processed[numerical_cols_to_scale])

    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    return df, df_processed, X_train, X_test, y_train, y_test


with st.spinner('Loading and preprocessing data...'):
    df, df_processed, X_train, X_test, y_train, y_test = load_and_preprocess_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview",
    "1. Overall Churn Distribution",
    "2. Churn Distribution by Gender",
    "3. Churn Distribution by Partner Status",
    "4. Churn Distribution by Dependents Status",
    "5. Churn Distribution by Senior Citizen Status",
    "6. Churn Distribution by Phone Service",
    "7. Churn Distribution by Multiple Lines Service",
    "8. Churn Distribution by Internet Service (General)",
    "9. Churn Distribution by Online Security Service",
    "10. Churn Distribution by Online Backup Service",
    "11. Churn Distribution by Device Protection Service",
    "12. Churn Distribution by Tech Support Service",
    "13. Churn Distribution by Streaming TV Service",
    "14. Churn Distribution by Streaming Movies Service",
    "15. Churn Distribution by Contract Type",
    "16. Churn Distribution by Paperless Billing",
    "17. Payment Method Distribution (Overall)",
    "18. Churn Distribution by Payment Method",
    "19. Churn Distribution w.r.t. Internet Service and Gender",
    "20. Detailed Numerical Feature Analysis - Tenure vs Churn",
    "21. Detailed Numerical Feature Analysis - Monthly Charges vs Churn",
    "22. Detailed Numerical Feature Analysis - Total Charges vs Churn",
    "23. Correlation Heatmap of All Features",
    "ML Model Performance",
    "Key Insights and Recommendations"
])

if page == "Project Overview":
    st.header("Project Overview")
    st.markdown("""
    This project focuses on analyzing customer churn in a telecommunications company. Customer churn, the phenomenon of customers discontinuing their service, is a critical business problem for Telco companies as it directly impacts revenue and profitability. By understanding the factors contributing to churn, the company can develop targeted strategies to retain valuable customers.

    ### Dataset
    The dataset used for this analysis is the `Telco Customer Churn` dataset, which contains information about 7,032 customers. It includes 20 features covering demographic information, services subscribed to, contract details, and payment information, along with a `Churn` target variable indicating whether a customer left the company.
       
    ### Problem Statement
    The primary goal is to identify factors influencing customer churn and build predictive models to accurately forecast which customers are likely to churn. This will enable the Telco company to proactively intervene with retention strategies.

    ### Objectives
    -   **Data Understanding and Cleaning**: Explore the dataset, identify data quality issues, and preprocess the data for analysis.
    -   **Exploratory Data Analysis (EDA) & Visualization**: Uncover patterns, relationships, and key drivers of churn through comprehensive data visualization.
    -   **Machine Learning Modeling**: Train and evaluate several classification models to predict customer churn.
    -   **Business Insights & Recommendations**: Translate analytical findings into actionable business strategies to reduce churn.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# --- Interactive Visualizations and Insights ---

elif page == "1. Overall Churn Distribution":
    st.header("1. Overall Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    churn_percentage = df['Churn'].value_counts(normalize=True) * 100

    fig = px.pie(names=churn_counts.index, values=churn_counts.values,
                 title='<b>Overall Churn Distribution</b>',
                 color_discrete_sequence=px.colors.sequential.Viridis_r,
                 hole=0.3)
    fig.update_traces(textinfo='percent+label',
                      marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(width=700, height=500,
                      annotations=[dict(text=f'Total Customers: {len(df)}', x=0.5, y=0.5, font_size=15, showarrow=False)])
    st.plotly_chart(fig)
    st.write(f"Overall Churn Rate: {churn_percentage['Yes']:.2f}%")
    st.write(f"Overall Non-Churn Rate: {churn_percentage['No']:.2f}%")
    st.info("**Business Insight**: The overall churn rate is approximately 26.58%, indicating a substantial loss of revenue. The dataset also shows a class imbalance that needs to be considered for modeling.")

elif page == "2. Churn Distribution by Gender":
    st.header("2. Churn Distribution by Gender")
    g_labels = ['Male', 'Female']
    c_labels = ['No', 'Yes']

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {
                        'type': 'domain'}]], horizontal_spacing=0.05)
    fig.add_trace(go.Pie(labels=g_labels,
                  values=df['gender'].value_counts(), name="Gender"), 1, 1)
    fig.add_trace(
        go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"), 1, 2)

    fig.update_traces(
        hole=.4, hoverinfo="label+percent+name", textfont_size=16)
    fig.update_layout(
        title_text="<b>Gender and Churn Distributions</b>",
        annotations=[dict(text='Gender', x=0.18, y=0.5, font_size=20, showarrow=False),
                     dict(text='Churn', x=0.82, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig)

    st.info("**Business Insight**: The analysis clearly shows a negligible difference in churn behavior between male (26.63% churn) and female (26.53% churn) customers. This implies that gender is not a significant factor for targeted churn prevention strategies. Retention efforts should focus on other, more influential factors.")

elif page == "3. Churn Distribution by Partner Status":
    st.header("3. Churn Distribution by Partner Status")
    churn_by_partner = df.groupby('Partner')['Churn'].value_counts(
        normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="Partner", barmode="group",
                       title="<b>Churn Distribution w.r.t. Partner Status</b>",
                       color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_partner)
    st.info("**Business Insight**: Customers without partners (32.98% churn) are significantly more prone to churn than those with a partner (19.70% churn). Targeted retention for single customers might be beneficial.")

elif page == "4. Churn Distribution by Dependents Status":
    st.header("4. Churn Distribution by Dependents Status")
    churn_by_dependents = df.groupby('Dependents')['Churn'].value_counts(
        normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="Dependents", barmode="group",
                       title="<b>Dependents distribution</b>",
                       color_discrete_map={"Yes": "#FF97FF", "No": "#AB63FA"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_dependents)
    st.info("**Business Insight**: Customers without dependents (31.39% churn) are more likely to churn compared to those with dependents (15.46% churn). This reinforces the idea that individuals with fewer ties might be less loyal. Targeted marketing and retention efforts could be tailored to these customers.")

elif page == "5. Churn Distribution by Senior Citizen Status":
    st.header("5. Churn Distribution by Senior Citizen Status")
    churn_by_senior = df.groupby('SeniorCitizen')['Churn'].value_counts(
        normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="SeniorCitizen",
                       title="<b>Churn distribution w.r.t. Senior Citizen</b>",
                       color_discrete_map={0: '#B6E880', 1: '#00CC96'})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_senior.rename(index={0: 'No', 1: 'Yes'}))
    st.info("**Business Insight**: Senior citizens have a significantly higher churn rate (41.68%) compared to non-senior citizens (23.65%). The company should investigate specific reasons for this and consider simplified plans or better customer support for this demographic.")

elif page == "6. Churn Distribution by Phone Service":
    st.header("6. Churn Distribution by Phone Service")
    churn_by_phone_service = df.groupby(
        'PhoneService')['Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="PhoneService",
                       title="<b>Churn distribution w.r.t. Phone Service</b>",
                       color_discrete_map={"Yes": '#00CC96', "No": '#B6E880'})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_phone_service)
    st.info("**Business Insight**: Almost all customers have phone service. Among the very few who don't, the churn rate (23.09%) is slightly lower than for those who do (26.92%). Phone service itself is not a primary churn driver.")

elif page == "7. Churn Distribution by Multiple Lines Service":
    st.header("7. Churn Distribution by Multiple Lines Service")
    churn_by_multiple_lines = df.groupby('MultipleLines')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="MultipleLines", barmode="group",
                       title="<b>Churn Distribution w.r.t. Multiple Lines Service</b>",
                       color_discrete_map={"Yes": "#00CC96", "No": "#B6E880", "No phone service": "#FECB52"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_multiple_lines)
    st.info("**Business Insight**: Customers with multiple lines (28.69% churn) churn slightly more than those with a single line (24.99% churn). Reviewing pricing structures or offering bundled discounts for multiple lines could enhance perceived value.")

elif page == "8. Churn Distribution by Internet Service (General)":
    st.header("8. Churn Distribution by Internet Service (General)")
    churn_by_internet_service = df.groupby('InternetService')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="InternetService", barmode="group",
                       title="<b>Churn Distribution w.r.t. Internet Service</b>",
                       color_discrete_map={
                           "DSL": "#FFA15A",
                           "Fiber optic": "#EF553B",
                           "No": "#636EFA"
                       })
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_internet_service)
    st.info("**Business Insight**: Fiber optic internet users show a strikingly high churn rate (41.88%). This is a critical area for concern. The company should investigate issues related to service reliability, speed consistency, customer support, or competitive pricing for Fiber optic.")

elif page == "9. Churn Distribution by Online Security Service":
    st.header("9. Churn Distribution by Online Security Service")
    churn_by_online_security = df.groupby('OnlineSecurity')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group",
                       title="<b>Churn w.r.t Online Security</b>",
                       color_discrete_map={"Yes": "#FF97FF", "No": "#AB63FA", "No internet service": "#FECB52"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_online_security)
    st.info("**Business Insight**: Customers who do not subscribe to Online Security services are significantly more likely to churn (42.14%) compared to those who do (14.34%). Promote and emphasize the benefits of online security.")

elif page == "10. Churn Distribution by Online Backup Service":
    st.header("10. Churn Distribution by Online Backup Service")
    churn_by_online_backup = df.groupby(
        'OnlineBackup')['Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="OnlineBackup", barmode="group",
                       title="<b>Churn Distribution w.r.t. Online Backup Service</b>",
                       color_discrete_map={"Yes": "#19D3F3", "No": "#E763A1", "No internet service": "#FF6692"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_online_backup)
    st.info("**Business Insight**: Customers who do not have 'Online Backup' service are significantly more likely to churn (39.93%) compared to those who do (21.68%). Actively promote the benefits of online backup and consider bundling it with internet services.")

elif page == "11. Churn Distribution by Device Protection Service":
    st.header("11. Churn Distribution by Device Protection Service")
    churn_by_device_protection = df.groupby('DeviceProtection')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="DeviceProtection", barmode="group",
                       title="<b>Churn Distribution w.r.t. Device Protection Service</b>",
                       color_discrete_map={"Yes": "#FF6692", "No": "#A0B1B6", "No internet service": "#636EFA"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_device_protection)
    st.info("**Business Insight**: Customers without 'Device Protection' service show a higher churn rate (39.10%) compared to those who have it (22.51%). Emphasize the value of device protection services in marketing and sales efforts.")

elif page == "12. Churn Distribution by Tech Support Service":
    st.header("12. Churn Distribution by Tech Support Service")
    churn_by_tech_support = df.groupby(
        'TechSupport')['Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="TechSupport", barmode="group",
                       title="<b>Churn distribution w.r.t. TechSupport</b>",
                       color_discrete_map={"Yes": "#19D3F3", "No": "#E763A1", "No internet service": "#FF6692"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_tech_support)
    st.info("**Business Insight**: Customers who do not subscribe to Tech Support services are significantly more likely to churn (42.06%) compared to those who do (14.33%). Prioritize providing robust and accessible tech support.")

elif page == "13. Churn Distribution by Streaming TV Service":
    st.header("13. Churn Distribution by Streaming TV Service")
    churn_by_streaming_tv = df.groupby(
        'StreamingTV')['Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="StreamingTV", barmode="group",
                       title="<b>Churn Distribution w.r.t. Streaming TV Service</b>",
                       color_discrete_map={"Yes": "#3DCCFF", "No": "#8A2BE2", "No internet service": "#008080"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_streaming_tv)
    st.info("**Business Insight**: Customers who do not subscribe to 'Streaming TV' have a slightly higher churn rate (33.52%) than those who do (30.08%). While it adds value, it's not as strong a churn driver as security or tech support. Could be used in broader entertainment bundles.")

elif page == "14. Churn Distribution by Streaming Movies Service":
    st.header("14. Churn Distribution by Streaming Movies Service")
    churn_by_streaming_movies = df.groupby('StreamingMovies')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="StreamingMovies", barmode="group",
                       title="<b>Churn Distribution w.r.t. Streaming Movies Service</b>",
                       color_discrete_map={"Yes": "#FFD700", "No": "#DAA520", "No internet service": "#B8860B"})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_streaming_movies)
    st.info("**Business Insight**: Similar to Streaming TV, customers without 'Streaming Movies' churn slightly more (33.32%) than those with the service (30.13%). It's a value-added amenity, not a primary churn deterrent. Integrate into comprehensive entertainment packages.")

elif page == "15. Churn Distribution by Contract Type":
    st.header("15. Churn Distribution by Contract Type")
    churn_by_contract = df.groupby('Contract')['Churn'].value_counts(
        normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="Contract", barmode="group",
                       title="<b>Customer contract distribution</b>")
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_contract)
    st.info("**Business Insight**: Customers on month-to-month contracts exhibit a significantly higher churn rate (42.71%) compared to those on one-year (11.25%) or two-year contracts (3.66%). Actively incentivize customers to opt for longer-term contracts with discounts or rewards.")

elif page == "16. Churn Distribution by Paperless Billing":
    st.header("16. Churn Distribution by Paperless Billing")
    churn_by_paperless_billing = df.groupby('PaperlessBilling')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="PaperlessBilling",
                       title="<b>Churn distribution w.r.t. Paperless Billing</b>",
                       color_discrete_map={"Yes": '#FFA15A', "No": '#00CC96'})
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_paperless_billing)
    st.info("**Business Insight**: Customers who opt for paperless billing show a higher propensity to churn (33.73%) compared to those who do not (16.33%). The company should investigate the customer experience with paperless billing to ensure it's seamless and doesn't inadvertently contribute to churn.")

elif page == "17. Payment Method Distribution (Overall)":
    st.header("17. Payment Method Distribution (Overall)")
    labels_pm = df['PaymentMethod'].unique()
    values_pm = df['PaymentMethod'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=labels_pm, values=values_pm, hole=.3)])
    fig.update_layout(title_text="<b>Payment Method Distribution</b>")
    st.plotly_chart(fig)
    st.dataframe(values_pm)
    st.info("**Business Insight**: 'Electronic check' is the most popular payment method. The company offers diversified payment options, which customers appreciate.")

elif page == "18. Churn Distribution by Payment Method":
    st.header("18. Churn Distribution by Payment Method")
    churn_by_payment_method = df.groupby('PaymentMethod')[
        'Churn'].value_counts(normalize=True).unstack() * 100
    fig = px.histogram(df, x="Churn", color="PaymentMethod",
                       title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
    fig.update_layout(width=700, height=500, bargap=0.1)
    st.plotly_chart(fig)
    st.dataframe(churn_by_payment_method)
    st.info("**Business Insight**: A disproportionately high number of customers using electronic checks (45.33% churn) tend to churn. Investigate why and incentivize switching to more stable payment methods like bank transfer or credit card automation.")

elif page == "19. Churn Distribution w.r.t. Internet Service and Gender":
    st.header("19. Churn Distribution w.r.t. Internet Service and Gender")
    # Re-using logic from original notebook, but will display a more aggregated view or a specific breakdown as Plotly bar charts
    fig = go.Figure()

    churn_no_female_dsl = df[(df["gender"] == "Female") & (
        df["InternetService"] == "DSL") & (df["Churn"] == "No")].shape[0]
    churn_no_male_dsl = df[(df["gender"] == "Male") & (
        df["InternetService"] == "DSL") & (df["Churn"] == "No")].shape[0]
    churn_yes_female_dsl = df[(df["gender"] == "Female") & (
        df["InternetService"] == "DSL") & (df["Churn"] == "Yes")].shape[0]
    churn_yes_male_dsl = df[(df["gender"] == "Male") & (
        df["InternetService"] == "DSL") & (df["Churn"] == "Yes")].shape[0]

    churn_no_female_fiber = df[(df["gender"] == "Female") & (
        df["InternetService"] == "Fiber optic") & (df["Churn"] == "No")].shape[0]
    churn_no_male_fiber = df[(df["gender"] == "Male") & (
        df["InternetService"] == "Fiber optic") & (df["Churn"] == "No")].shape[0]
    churn_yes_female_fiber = df[(df["gender"] == "Female") & (
        df["InternetService"] == "Fiber optic") & (df["Churn"] == "Yes")].shape[0]
    churn_yes_male_fiber = df[(df["gender"] == "Male") & (
        df["InternetService"] == "Fiber optic") & (df["Churn"] == "Yes")].shape[0]

    churn_no_female_no_internet = df[(df["gender"] == "Female") & (
        df["InternetService"] == "No") & (df["Churn"] == "No")].shape[0]
    churn_no_male_no_internet = df[(df["gender"] == "Male") & (
        df["InternetService"] == "No") & (df["Churn"] == "No")].shape[0]
    churn_yes_female_no_internet = df[(df["gender"] == "Female") & (
        df["InternetService"] == "No") & (df["Churn"] == "Yes")].shape[0]
    churn_yes_male_no_internet = df[(df["gender"] == "Male") & (
        df["InternetService"] == "No") & (df["Churn"] == "Yes")].shape[0]

    fig.add_trace(go.Bar(
        x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
            ["Female", "Male", "Female", "Male"]],
        y=[churn_no_female_dsl, churn_no_male_dsl,
            churn_yes_female_dsl, churn_yes_male_dsl],
        name='DSL',
    ))

    fig.add_trace(go.Bar(
        x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
            ["Female", "Male", "Female", "Male"]],
        y=[churn_no_female_fiber, churn_no_male_fiber,
            churn_yes_female_fiber, churn_yes_male_fiber],
        name='Fiber optic',
    ))

    fig.add_trace(go.Bar(
        x=[['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
            ["Female", "Male", "Female", "Male"]],
        y=[churn_no_female_no_internet, churn_no_male_no_internet,
            churn_yes_female_no_internet, churn_yes_male_no_internet],
        name='No Internet',
    ))

    fig.update_layout(
        title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>", barmode='group')
    st.plotly_chart(fig)
    st.info("**Business Insight**: This confirms that Fiber optic internet service customers have a significantly higher churn rate (41.88% overall for Fiber Optic users) compared to DSL (18.96% churn) or no internet service (7.49% churn), regardless of gender. This reinforces that gender is not a primary churn driver. Prioritize investigating Fiber optic service issues.")

elif page == "20. Detailed Numerical Feature Analysis - Tenure vs Churn":
    st.header("20. Detailed Numerical Feature Analysis - Tenure vs Churn")
    fig = px.box(df, x='Churn', y='tenure')
    fig.update_yaxes(title_text='Tenure (Months)')
    fig.update_xaxes(title_text='Churn')
    fig.update_layout(autosize=True, width=750, height=600,
                      title_font=dict(size=25, family='Courier'),
                      title='<b>Tenure vs Churn</b>')
    st.plotly_chart(fig)
    st.info("**Business Insight**: Customers with very short tenures (median for churned customers is ~10 months) are much more likely to churn compared to non-churning customers (~38 months). Invest heavily in robust onboarding and early engagement strategies.")

elif page == "21. Detailed Numerical Feature Analysis - Monthly Charges vs Churn":
    st.header("21. Detailed Numerical Feature Analysis - Monthly Charges vs Churn")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['Churn'] == 'No']['MonthlyCharges'],
                  histnorm='probability density', name='Not Churn', marker_color='red', opacity=0.6))
    fig.add_trace(go.Histogram(x=df[df['Churn'] == 'Yes']['MonthlyCharges'],
                  histnorm='probability density', name='Churn', marker_color='blue', opacity=0.6))

    fig.update_layout(
        barmode='overlay',
        title='<b>Distribution of Monthly Charges by Churn</b>',
        xaxis_title='Monthly Charges',
        yaxis_title='Density'
    )
    st.plotly_chart(fig)
    st.info("**Business Insight**: The density of churning customers peaks at higher monthly charges (around $80-$100) compared to non-churning customers. This suggests customers paying higher fees might have higher expectations. Review pricing strategy and ensure perceived value aligns with cost.")

elif page == "22. Detailed Numerical Feature Analysis - Total Charges vs Churn":
    st.header("22. Detailed Numerical Feature Analysis - Total Charges vs Churn")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['Churn'] == 'No']['TotalCharges'],
                  histnorm='probability density', name='Not Churn', marker_color='gold', opacity=0.6))
    fig.add_trace(go.Histogram(x=df[df['Churn'] == 'Yes']['TotalCharges'],
                  histnorm='probability density', name='Churn', marker_color='green', opacity=0.6))

    fig.update_layout(
        barmode='overlay',
        title='<b>Distribution of Total Charges by Churn</b>',
        xaxis_title='Total Charges',
        yaxis_title='Density'
    )
    st.plotly_chart(fig)
    st.info("**Business Insight**: Churning customers tend to have lower total charges (peaking around $100-$500), which aligns with shorter tenure. This emphasizes the importance of early customer engagement and value demonstration to increase lifetime value.")

elif page == "23. Correlation Heatmap of All Features":
    st.header("23. Correlation Heatmap of All Features")
    # Recreate df_factorized for correlation heatmap
    df_factorized = df.apply(lambda x: pd.factorize(x)[0])
    corr = df_factorized.corr()

    fig = px.imshow(corr,
                    text_auto=True, aspect="auto",
                    title='<b>Correlation Heatmap of All Features</b>',
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1])
    fig.update_layout(width=1000, height=800)
    st.plotly_chart(fig)
    st.info("**Business Insight**: Strong positive correlation with Churn: Month-to-month contract (0.40), InternetService_Fiber optic (0.30), PaperlessBilling (0.19), and PaymentMethod_Electronic check (0.29). Strong negative correlation with Churn: Tenure (-0.35), Contract_Two year (-0.30), OnlineSecurity (-0.27), TechSupport (-0.28). This heatmap visually summarizes the relationships, confirming many of the individual insights.")

elif page == "ML Model Performance":
    st.header("ML Model Performance")
    st.write("Here we evaluate the performance of three machine learning models: Logistic Regression, Random Forest, and XGBoost Classifier.")

    st.subheader("Logistic Regression Model")
    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    log_reg_model.fit(X_train, y_train)
    y_pred_lr = log_reg_model.predict(X_test)
    y_proba_lr = log_reg_model.predict_proba(X_test)[:, 1]

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    st.write(f"Precision (Churn): {precision_score(y_test, y_pred_lr):.4f}")
    st.write(f"Recall (Churn): {recall_score(y_test, y_pred_lr):.4f}")
    st.write(f"F1-Score (Churn): {f1_score(y_test, y_pred_lr):.4f}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_proba_lr):.4f}")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred_lr), index=[
                 'Actual No Churn', 'Actual Churn'], columns=['Predicted No Churn', 'Predicted Churn']))
    st.code(classification_report(y_test, y_pred_lr))
    st.info("**Business Insight**: Logistic Regression provides a decent baseline with 80% accuracy and a good ROC AUC of 0.84. Precision (0.64) is better than recall (0.57) for the churn class, which needs consideration given the class imbalance. For churn prediction, a higher recall is often desired.")

    st.subheader("Random Forest Classifier")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    st.write(f"Precision (Churn): {precision_score(y_test, y_pred_rf):.4f}")
    st.write(f"Recall (Churn): {recall_score(y_test, y_pred_rf):.4f}")
    st.write(f"F1-Score (Churn): {f1_score(y_test, y_pred_rf):.4f}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_proba_rf):.4f}")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred_rf), index=[
                 'Actual No Churn', 'Actual Churn'], columns=['Predicted No Churn', 'Predicted Churn']))
    st.code(classification_report(y_test, y_pred_rf))
    st.info("**Business Insight**: Random Forest achieved slightly lower accuracy (79.3%) and recall (50%) for the churn class compared to Logistic Regression. Further hyperparameter tuning might improve its performance.")

elif page == "Key Insights and Recommendations":
    st.header("Key Insights and Recommendations")
    st.markdown("""
    Based on the comprehensive analysis, here are the key recommendations for the Telco company to mitigate customer churn:

    -   **Retention Programs for New Customers**: Focus heavily on robust onboarding and early engagement strategies for new customers, as those with a median tenure of only **10 months** are significantly more likely to churn compared to loyal customers with a median tenure of **38 months**. Promptly address any issues to convert new users into long-term subscribers.

    -   **Promote Value-Added Services**: Actively encourage customers to subscribe to services like **Online Security**, **Online Backup**, **Device Protection**, and **Tech Support**. Customers without these services churn at rates significantly higher (e.g., **42.14%** for no Online Security, **42.06%** for no Tech Support) compared to those who have them (e.g., **14.34%** for Online Security, **14.33%** for Tech Support). Bundling these services can enhance customer loyalty.

    -   **Incentivize Longer Contracts**: Implement strategies to encourage customers to switch from month-to-month contracts (which have a high churn rate of **42.71%**) to longer-term one-year (**11.25%** churn) or two-year contracts (**3.66%** churn). This could involve offering discounts, exclusive benefits, or loyalty rewards for committing to longer periods.

    -   **Address Fiber Optic Service Issues**: Conduct a thorough investigation into customer satisfaction and potential pain points for **Fiber optic** internet users. This segment has a significantly higher churn rate of **41.88%** compared to DSL (**18.96%** churn) and customers with no internet service (**7.49%** churn). Improving service quality or pricing transparency for Fiber optic can yield substantial retention benefits.

    -   **Optimize Payment Method Experience**: Investigate why customers using **Electronic Check** have the highest churn rate at **45.33%**. This is much higher than other automatic payment methods like Bank Transfer (**16.71%** churn) or Credit Card (**16.01%** churn). Incentivize these customers to adopt more stable payment methods or address underlying issues with the electronic check process.

    -   **Engage Senior Citizens with Tailored Offers**: Develop specific retention programs or simplified service offerings for senior citizens, as they exhibit a much higher churn rate of **41.68%** compared to non-senior citizens at **23.65%**. Understanding their unique needs and providing relevant support can improve their loyalty.

    -   **Review Higher Monthly Charges**: Analyze customer feedback and value perception for higher-tier plans. Customers with higher monthly charges (peaking around $80-$100 for churners) are more prone to churn, suggesting that perceived value might not align with the cost. Re-evaluate pricing strategies or enhance the value proposition for these plans.
    """)

st.sidebar.markdown("--- ")
st.sidebar.info("Dashboard created with Streamlit and Plotly.")
