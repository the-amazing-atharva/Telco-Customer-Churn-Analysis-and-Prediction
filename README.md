# Telco Customer Churn Analysis

## 1. Project Overview

This project focuses on analyzing customer churn in a telecommunications company. Customer churn, the phenomenon of customers discontinuing their service, is a critical business problem for Telco companies as it directly impacts revenue and profitability. By understanding the factors contributing to churn, the company can develop targeted strategies to retain valuable customers.

## 2. Dataset

The dataset used for this analysis is the `Telco Customer Churn` dataset, which contains information about 7,032 customers. It includes 20 features covering demographic information, services subscribed to, contract details, and payment information, along with a `Churn` target variable indicating whether a customer left the company.

<img width="1700" height="367" alt="image" src="https://github.com/user-attachments/assets/31df09fd-7574-49cb-89de-da00e4aa66e0" />

## 3. Problem Statement

The primary goal is to identify factors influencing customer churn and build predictive models to accurately forecast which customers are likely to churn. This will enable the Telco company to proactively intervene with retention strategies.

## 4. Objectives

- **Data Understanding and Cleaning**: Explore the dataset, identify data quality issues, and preprocess the data for analysis.
- **Exploratory Data Analysis (EDA) & Visualization**: Uncover patterns, relationships, and key drivers of churn through comprehensive data visualization.
- **Machine Learning Modeling**: Train and evaluate several classification models to predict customer churn.
- **Business Insights & Recommendations**: Translate analytical findings into actionable business strategies to reduce churn.

## 5. Methodology and Approach

### 5.1 Data Understanding & Cleaning

- Initial inspection using `df.head()`, `df.info()`, `df.describe()`, and `df.shape` revealed data types and basic statistics.
- The `TotalCharges` column, initially an `object` type, was converted to `numeric`, and 11 rows with missing values (represented as spaces) were dropped.
- The `customerID` column was removed as it's not relevant for prediction.
- The target variable `Churn` showed an imbalance: **73.4%** 'No' (non-churn) and **26.6%** 'Yes' (churn).

<img width="448" height="606" alt="image" src="https://github.com/user-attachments/assets/6c152005-adb1-4ff5-8d8f-f6d964ba13c2" />

### 5.2 Exploratory Data Analysis (EDA) & Visualization

Extensive visualization was performed to understand the relationship between various features and churn:

- **Overall Churn Distribution**: The overall churn rate is **26.58%**, highlighting a significant retention challenge.
- **Gender**: No significant difference in churn rates between male (**26.63%**) and female (**26.53%**) customers.
- **Partner & Dependents**: Customers without partners (**32.98%** churn) or dependents (**31.39%** churn) are more likely to churn.
- **Senior Citizens**: Senior citizens have a significantly higher churn rate (**41.68%**) compared to non-senior citizens (**23.65%**).
- **Phone Service & Multiple Lines**: Phone service presence itself isn't a strong indicator. Customers with multiple lines (**28.69%** churn) churn slightly more than those with single lines (**24.99%** churn).
- **Internet Service**: **Fiber optic** internet customers show a strikingly high churn rate (**41.88%**), much higher than DSL (**18.96%**) or no internet service (**7.49%**).
- **Security & Support Services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport)**: Lack of these services is strongly associated with higher churn (e.g., **42.14%** for no Online Security, **42.06%** for no Tech Support), indicating they are crucial retention factors.
- **Streaming Services (TV, Movies)**: While contributing to overall satisfaction, their presence or absence is not as strong a churn driver as security/support services.
- **Contract Type**: **Month-to-month contracts** have a significantly higher churn rate (**42.71%**) compared to one-year (**11.25%**) and two-year (**3.66%**) contracts.
- **Paperless Billing**: Customers with paperless billing show higher churn (**33.73%**) than those without (**16.33%**).
- **Payment Method**: **Electronic Check** users have the highest churn rate (**45.33%**), significantly higher than other automatic payment methods (e.g., Credit Card: **16.01%**).
- **Tenure**: New customers (median tenure of ~**10 months** for churners) are much more prone to churn; loyalty increases with tenure (median ~**38 months** for non-churners).
- **Monthly Charges**: Customers with higher monthly charges (peaking around **$80-$100** for churners) tend to churn more.
- **Total Charges**: Churning customers generally have lower total charges (peaking around **$100-$500**), consistent with shorter tenure.
- **Correlation Heatmap**: Confirmed strong positive correlations between churn and `Month-to-month contract`, `InternetService_Fiber optic`, `PaperlessBilling`, and `PaymentMethod_Electronic check`. Strong negative correlations were observed with `Tenure`, `Contract_Two year`, `OnlineSecurity`, and `TechSupport`.

<img width="1790" height="1801" alt="image" src="https://github.com/user-attachments/assets/052e2033-2b39-4d9b-b817-0aa519f2ea4a" />

<img width="1885" height="944" alt="image" src="https://github.com/user-attachments/assets/0b912c06-47f2-446d-bda7-0ef97f027fce" />

### 5.3 Data Preprocessing for Machine Learning

- **Encoding**: Binary categorical features (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`, `Churn`) were mapped to 0/1. Multi-class features (`InternetService`, `Contract`, `PaymentMethod`) were one-hot encoded.
- **Scaling**: Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler`.
- **Splitting**: Data was split into 80% training and 20% testing sets, stratified by the `Churn` variable to maintain class distribution.

### 5.4 Machine Learning Modeling

Three classification models were trained and evaluated:

1.  **Logistic Regression** (Baseline)
    - Accuracy: **0.8031**
    - Precision (Churn): **0.6483**
    - Recall (Churn): **0.5668**
    - F1-Score (Churn): **0.6049**
    - ROC AUC: **0.8362**

2.  **Random Forest Classifier**
    - Accuracy: **0.7932**
    - Precision (Churn): **0.6416**
    - Recall (Churn): **0.5027**
    - F1-Score (Churn): **0.5637**
    - ROC AUC: **0.8229**

3.  **XGBoost Classifier**
    - Accuracy: **0.7711**
    - Precision (Churn): **0.5760**
    - Recall (Churn): **0.5267**
    - F1-Score (Churn): **0.5503**
    - ROC AUC: **0.8114**

**Model Comparison**: In this specific execution, **Logistic Regression** performed the best across key metrics (Accuracy, Recall, F1-Score, ROC AUC) for the churn class. This suggests it provides a strong baseline for churn prediction given the dataset and preprocessing steps.

<img width="576" height="497" alt="image" src="https://github.com/user-attachments/assets/e9f8c8a0-e144-410d-b15a-4848f613ca8a" />

### 📊 5.5 Streamlit Dashboard for Interactive Visualisations

This Streamlit dashboard offers an interactive platform for a comprehensive analysis of customer churn in a telecommunications company. It visualizes key churn drivers, showcases machine learning model performance, and outlines actionable recommendations, all based on the in-depth analysis conducted in this notebook.

#### 🚀 Features

- **Project Overview**: Introduction to the problem, objectives, and a dataset preview.
- **23 Interactive Visualizations**: Explore churn distribution across various demographic, service, and contract features.
- **Machine Learning Performance**: Review and compare Logistic Regression, Random Forest, and XGBoost models for churn prediction.
- **Key Recommendations**: Data-driven strategies to mitigate customer churn.

#### 🛠️ Setup and Run (for the Streamlit App)

To run this dashboard locally, follow these steps:

1.  **Prerequisites**: Ensure you have Python (3.7+) installed on your system.

2.  **Install Dependencies**: (It's recommended to use a virtual environment)

    First, create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

    Then, install the required Python libraries:

    ```bash
    pip install pandas numpy scikit-learn xgboost plotly seaborn streamlit
    ```

3.  **Save the Streamlit App**: Save the Python code for the dashboard (e.g., in a file named `streamlit_app.py`) on your local machine. Ensure your `Telco-Customer-Churn.csv` dataset is in the same directory as this `streamlit_app.py` file, or update the data loading path within the Streamlit app code.

4.  **Run the App**: Open your terminal or command prompt. Navigate to the directory where you saved `streamlit_app.py` and execute the following command:

    ```bash
    streamlit run streamlit_app.py
    ```

    The dashboard will then automatically open in your default web browser (typically at `http://localhost:8501`).

## 6. Key Insights and Recommendations

Based on the comprehensive analysis, here are the key recommendations for the Telco company to mitigate customer churn:

- **Retention Programs for New Customers**: Focus heavily on robust onboarding and early engagement strategies for new customers, as those with a median tenure of only **10 months** are significantly more likely to churn compared to loyal customers with a median tenure of **38 months**. Promptly address any issues to convert new users into long-term subscribers.

- **Promote Value-Added Services**: Actively encourage customers to subscribe to services like **Online Security**, **Online Backup**, **Device Protection**, and **Tech Support**. Customers without these services churn at rates significantly higher (e.g., **42.14%** for no Online Security, **42.06%** for no Tech Support) compared to those who have them (e.g., **14.34%** for Online Security, **14.33%** for Tech Support). Bundling these services can enhance customer loyalty.

- **Incentivize Longer Contracts**: Implement strategies to encourage customers to switch from month-to-month contracts (which have a high churn rate of **42.71%**) to longer-term one-year (**11.25%** churn) or two-year contracts (**3.66%** churn). This could involve offering discounts, exclusive benefits, or loyalty rewards for committing to longer periods.

- **Address Fiber Optic Service Issues**: Conduct a thorough investigation into customer satisfaction and potential pain points for **Fiber optic** internet users. This segment has a significantly higher churn rate of **41.88%** compared to DSL (**18.96%** churn) and customers with no internet service (**7.49%** churn). Improving service quality or pricing transparency for Fiber optic can yield substantial retention benefits.

- **Optimize Payment Method Experience**: Investigate why customers using **Electronic Check** have the highest churn rate at **45.33%**. This is much higher than other automatic payment methods like Bank Transfer (**16.71%** churn) or Credit Card (**16.01%** churn). Incentivize these customers to adopt more stable payment methods or address underlying issues with the electronic check process.

- **Engage Senior Citizens with Tailored Offers**: Develop specific retention programs or simplified service offerings for senior citizens, as they exhibit a much higher churn rate of **41.68%** compared to non-senior citizens at **23.65%**. Understanding their unique needs and providing relevant support can improve their loyalty.

- **Review Higher Monthly Charges**: Analyze customer feedback and value perception for higher-tier plans. Customers with higher monthly charges (peaking around $80-$100 for churners) are more prone to churn, suggesting that perceived value might not align with the cost. Re-evaluate pricing strategies or enhance the value proposition for these plans.

## 7. Conclusion

This analysis provides a strong foundation for understanding customer churn, identifying its root causes through data-driven insights, and developing targeted retention strategies. The predictive models can help pinpoint at-risk customers, allowing the Telco company to implement timely interventions and improve customer lifetime value.
