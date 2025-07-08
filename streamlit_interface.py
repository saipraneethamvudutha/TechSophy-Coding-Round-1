import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from joblib import load
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Insurance Claim Fraud Detection", layout="wide")

# Title and description
st.title("Insurance Claim Fraud Detection System")
st.markdown("""
This application uses pre-trained XGBoost and Isolation Forest models to detect potential insurance claim fraud. You can:
1. **Upload a CSV or Excel file** for batch predictions with visualizations.
2. **Enter claim details manually** for real-time predictions.

---

### Required Data Format
**File Types**: CSV (`.csv`) or Excel (`.xls`, `.xlsx`)

**Required Columns (16 input features)**:
- **Numerical**:
  - `premium_amount`: Float (e.g., 1000.0)
  - `claim_amount`: Float (e.g., 5000.0)
  - `age`: Integer (e.g., 35)
  - `tenure`: Float (e.g., 2.0)
  - `no_of_family_members`: Integer (e.g., 3)
  - `incident_hour_of_the_day`: Integer (0–23, e.g., 14)
  - `any_injury`: Binary (0 or 1)
  - `police_report_available`: Binary (0 or 1)
- **Categorical** (strings):
  - `employment_status`: e.g., "Employed", "Unemployed", "Retired"
  - `house_type`: e.g., "Own", "Rent", "Other"
  - `social_class`: e.g., "High", "Middle", "Low"
  - `incident_severity`: e.g., "Minor", "Major", "Total Loss"
  - `authority_contacted`: e.g., "Police", "None", "Ambulance"
  - `insurance_type`: e.g., "Auto", "Home", "Health"
  - `customer_education_level`: e.g., "High School", "Bachelor", "Master", "PhD"
  - `risk_segmentation`: e.g., "Low", "Medium", "High"

**Derived Features** (calculated automatically):
- `claim_to_premium_ratio`: `claim_amount / (premium_amount + 1)`
- `claim_per_person`: `claim_amount / (no_of_family_members + 1)`

**Notes**:
- No missing values; fill with 0 or appropriate defaults.
- Categorical columns must match training data categories to ensure consistent encoding.
- `claim_status` (target) is **not** required for predictions.
- **File Size Limit**: Ensure files are reasonable in size (e.g., <50MB) to avoid memory issues.

**Example CSV Format**:
```
premium_amount,claim_amount,age,tenure,no_of_family_members,incident_hour_of_the_day,employment_status,house_type,social_class,incident_severity,authority_contacted,insurance_type,customer_education_level,risk_segmentation,any_injury,police_report_available
1000.0,5000.0,35,2.0,3,14,Employed,Own,High,Major,Police,Auto,Bachelor,Low,1,1
500.0,2000.0,45,5.0,2,9,Unemployed,Rent,Middle,Minor,Ambulance,Home,High School,Medium,0,0
```
""")

# Load pre-trained models
try:
    clf = XGBClassifier()
    clf.load_model('xgboost_model.json')
    iso = load('isolation_forest.joblib')
    st.success("Pre-trained XGBoost and Isolation Forest models loaded successfully.")
except FileNotFoundError:
    st.error("Model files not found. Ensure 'xgboost_model.json' and 'isolation_forest.joblib' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Define features
features = [
    'premium_amount', 'claim_amount', 'age', 'tenure', 'no_of_family_members',
    'incident_hour_of_the_day', 'employment_status', 'house_type', 'social_class',
    'incident_severity', 'authority_contacted', 'any_injury',
    'police_report_available', 'insurance_type', 'customer_education_level',
    'risk_segmentation', 'claim_to_premium_ratio', 'claim_per_person'
]

required_cols = [
    'premium_amount', 'claim_amount', 'age', 'tenure', 'no_of_family_members',
    'incident_hour_of_the_day', 'employment_status', 'house_type', 'social_class',
    'incident_severity', 'authority_contacted', 'any_injury',
    'police_report_available', 'insurance_type', 'customer_education_level',
    'risk_segmentation'
]

categorical_cols = [
    'employment_status', 'house_type', 'social_class', 'incident_severity',
    'authority_contacted', 'insurance_type', 'customer_education_level',
    'risk_segmentation'
]

# Sample dataset for download
sample_data = pd.DataFrame({
    'premium_amount': [1000.0, 500.0, 750.0, 1200.0, 800.0, 600.0, 1500.0, 900.0, 1100.0, 700.0],
    'claim_amount': [5000.0, 2000.0, 10000.0, 3000.0, 6000.0, 1500.0, 8000.0, 4000.0, 7000.0, 2500.0],
    'age': [35, 45, 28, 50, 40, 30, 60, 25, 38, 42],
    'tenure': [2.0, 5.0, 1.5, 7.0, 3.0, 2.5, 10.0, 1.0, 4.0, 6.0],
    'no_of_family_members': [3, 2, 1, 4, 2, 0, 3, 1, 2, 3],
    'incident_hour_of_the_day': [14, 9, 22, 12, 18, 6, 15, 20, 10, 8],
    'employment_status': ['Employed', 'Unemployed', 'Employed', 'Retired', 'Employed', 'Unemployed', 'Retired', 'Employed', 'Employed', 'Unemployed'],
    'house_type': ['Own', 'Rent', 'Rent', 'Own', 'Own', 'Rent', 'Own', 'Rent', 'Own', 'Rent'],
    'social_class': ['High', 'Middle', 'Low', 'High', 'Middle', 'Low', 'High', 'Middle', 'High', 'Middle'],
    'incident_severity': ['Major', 'Minor', 'Total Loss', 'Minor', 'Major', 'Minor', 'Total Loss', 'Major', 'Minor', 'Minor'],
    'authority_contacted': ['Police', 'Ambulance', 'Police', 'None', 'Police', 'None', 'Ambulance', 'Police', 'None', 'Ambulance'],
    'insurance_type': ['Auto', 'Home', 'Auto', 'Health', 'Auto', 'Home', 'Health', 'Auto', 'Home', 'Health'],
    'customer_education_level': ['Bachelor', 'High School', 'Master', 'PhD', 'Bachelor', 'High School', 'Master', 'Bachelor', 'PhD', 'High School'],
    'risk_segmentation': ['Low', 'Medium', 'High', 'Low', 'Medium', 'Low', 'High', 'Medium', 'Low', 'Medium'],
    'any_injury': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    'police_report_available': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
})

# Function to preprocess data
def preprocess_data(df):
    df = df.copy()
    # Calculate derived features
    df['claim_to_premium_ratio'] = df['claim_amount'] / (df['premium_amount'] + 1)
    df['family_size'] = df['no_of_family_members'] + 1
    df['claim_per_person'] = df['claim_amount'] / df['family_size']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Fill missing values
    df.fillna(0, inplace=True)
    
    return df[features]

# Function to apply color coding to Fraud_Probability or Anomaly_Score
def color_score(val, col_name, thresholds=None):
    if isinstance(val, float):
        if col_name == 'Fraud_Probability':
            if val > 0.66:
                return 'background-color: red'
            elif val > 0.33:
                return 'background-color: yellow'
            else:
                return 'background-color: green'
        elif col_name == 'Anomaly_Score' and thresholds is not None:
            low_threshold, high_threshold = thresholds
            if val > high_threshold:
                return 'background-color: red'
            elif val > low_threshold:
                return 'background-color: yellow'
            else:
                return 'background-color: green'
    return ''

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose Model", ["XGBoost", "Isolation Forest"])

# File upload section
st.header("Upload Data for Batch Predictions")
st.info("Upload a CSV or Excel file for batch predictions. For XGBoost, 'Fraud_Probability' is color-coded: red (>0.66), yellow (0.33–0.66), green (<0.33). For Isolation Forest, 'Anomaly_Score' is color-coded: red (top 33%), yellow (middle 33%), green (bottom 33%). Visualizations display risk scores, claim status, priority levels (XGBoost only), and feature importance (XGBoost only).")
st.download_button(
    label="Download Sample Dataset (insurance_data.csv)",
    data=sample_data.to_csv(index=False),
    file_name="insurance_data.csv",
    mime="text/csv",
)
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Preprocess data
            X = preprocess_data(df)
            
            # Make predictions and visualize
            if model_choice == "XGBoost":
                st.subheader("XGBoost Predictions")
                proba = clf.predict_proba(X)[:, 1]
                predictions = clf.predict(X)
                results = df.copy()
                results['Fraud_Probability'] = proba
                results['Prediction'] = pd.Series(predictions).map({0: 'Not Fraud', 1: 'Fraud'})
                results['Priority'] = pd.qcut(proba, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                
                # Display results with color-coded Fraud_Probability
                st.write("### Prediction Results")
                st.dataframe(
                    results[['Fraud_Probability', 'Prediction', 'Priority'] + required_cols]
                    .style.apply(lambda x: [color_score(val, 'Fraud_Probability') for val in x], subset=['Fraud_Probability'])
                    .format({'Fraud_Probability': '{:.3f}'})
                )
                
                # Visualizations
                st.subheader("Visual Insights")
                
                # Risk Score Distribution
                fig_risk = px.histogram(
                    results, x='Fraud_Probability', nbins=30, title="Distribution of Risk Scores",
                    labels={'Fraud_Probability': 'Fraud Probability'}, color_discrete_sequence=['purple']
                )
                fig_risk.update_layout(showlegend=False, bargap=0.1)
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Claim Status Distribution
                fig_status = px.histogram(
                    results, x='Prediction', title="Claim Status Distribution",
                    labels={'Prediction': 'Predicted Status'}, color='Prediction',
                    color_discrete_map={'Not Fraud': '#00CC96', 'Fraud': '#EF553B'}
                )
                fig_status.update_layout(showlegend=False, bargap=0.2)
                st.plotly_chart(fig_status, use_container_width=True)
                
                # Priority Breakdown
                fig_priority = px.histogram(
                    results, x='Priority', title="Investigation Priority Levels",
                    labels={'Priority': 'Priority Level'}, color='Priority',
                    color_discrete_map={'Low': '#00CC96', 'Medium': '#FFA15A', 'High': '#EF553B'}
                )
                fig_priority.update_layout(showlegend=False, bargap=0.2)
                st.plotly_chart(fig_priority, use_container_width=True)
                
                # Feature Importance
                try:
                    importances = clf.feature_importances_
                    feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)
                    fig_importance = px.bar(
                        feat_importance, title="Feature Importances from XGBoost",
                        labels={'value': 'Importance', 'index': 'Feature'}, color_discrete_sequence=['teal']
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying feature importance: {str(e)}")
            
            else:  # Isolation Forest
                st.subheader("Isolation Forest Anomaly Detection")
                anomalies = iso.predict(X)
                scores = -iso.decision_function(X)  # Positive scores indicate anomalies
                results = df.copy()
                results['Anomaly_Score'] = scores
                results['Prediction'] = pd.Series(anomalies).map({1: 'Normal', -1: 'Potential Fraud'})
                
                # Calculate thresholds for anomaly scores (33rd and 66th percentiles)
                if len(scores) > 1:
                    low_threshold, high_threshold = np.percentile(scores, [33, 66])
                else:
                    low_threshold, high_threshold = 0, 0  # Default for single row
                
                # Display results with color-coded Anomaly_Score
                st.write("### Prediction Results")
                st.dataframe(
                    results[['Anomaly_Score', 'Prediction'] + required_cols]
                    .style.apply(lambda x: [color_score(val, 'Anomaly_Score', (low_threshold, high_threshold)) for val in x], subset=['Anomaly_Score'])
                    .format({'Anomaly_Score': '{:.3f}'})
                )
                
                # Visualizations
                st.subheader("Visual Insights")
                
                # Anomaly Score Distribution
                fig_risk = px.histogram(
                    results, x='Anomaly_Score', nbins=30, title="Distribution of Anomaly Scores",
                    labels={'Anomaly_Score': 'Anomaly Score'}, color_discrete_sequence=['purple']
                )
                fig_risk.update_layout(showlegend=False, bargap=0.1)
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Claim Status Distribution
                fig_status = px.histogram(
                    results, x='Prediction', title="Claim Status Distribution",
                    labels={'Prediction': 'Predicted Status'}, color='Prediction',
                    color_discrete_map={'Normal': '#00CC96', 'Potential Fraud': '#EF553B'}
                )
                fig_status.update_layout(showlegend=False, bargap=0.2)
                st.plotly_chart(fig_status, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Manual input section
st.header("Manual Data Input")
st.info("Enter claim details below. Ensure categorical inputs match training data categories (e.g., 'Employed', 'Own', 'High') for accurate predictions.")

with st.form("manual_input"):
    st.write("Enter Claim Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        premium_amount = st.number_input("Premium Amount", min_value=0.0, value=1000.0, step=0.1)
        claim_amount = st.number_input("Claim Amount", min_value=0.0, value=5000.0, step=0.1)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (years)", min_value=0.0, value=2.0, step=0.1)
        no_of_family_members = st.number_input("No. of Family Members", min_value=0, value=3)
        incident_hour = st.number_input("Incident Hour (0-23)", min_value=0, max_value=23, value=14)
    
    with col2:
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"], index=0)
        house_type = st.selectbox("House Type", ["Own", "Rent", "Other"], index=0)
        social_class = st.selectbox("Social Class", ["High", "Middle", "Low"], index=0)
        incident_severity = st.selectbox("Incident Severity", ["Minor", "Major", "Total Loss"], index=0)
        authority_contacted = st.selectbox("Authority Contacted", ["Police", "None", "Ambulance"], index=0)
    
    with col3:
        insurance_type = st.selectbox("Insurance Type", ["Auto", "Home", "Health"], index=0)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"], index=0)
        risk_segmentation = st.selectbox("Risk Segmentation", ["Low", "Medium", "High"], index=0)
        any_injury = st.selectbox("Any Injury", [0, 1], index=1)
        police_report = st.selectbox("Police Report Available", [0, 1], index=1)
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'premium_amount': [premium_amount],
            'claim_amount': [claim_amount],
            'age': [age],
            'tenure': [tenure],
            'no_of_family_members': [no_of_family_members],
            'incident_hour_of_the_day': [incident_hour],
            'employment_status': [employment_status],
            'house_type': [house_type],
            'social_class': [social_class],
            'incident_severity': [incident_severity],
            'authority_contacted': [authority_contacted],
            'insurance_type': [insurance_type],
            'customer_education_level': [education_level],
            'risk_segmentation': [risk_segmentation],
            'any_injury': [any_injury],
            'police_report_available': [police_report]
        })
        
        try:
            # Preprocess input
            X_input = preprocess_data(input_data)
            
            # Make predictions
            if model_choice == "XGBoost":
                st.subheader("XGBoost Prediction")
                proba = clf.predict_proba(X_input)[:, 1][0]
                prediction = clf.predict(X_input)[0]
                priority = 'High' if proba > 0.66 else 'Medium' if proba > 0.33 else 'Low'
                st.success(f"**Prediction**: {'Fraud' if prediction == 1 else 'Not Fraud'}")
                st.write(f"**Fraud Probability**: {proba:.3f}")
                st.write(f"**Priority**: {priority}")
            
            else:  # Isolation Forest
                st.subheader("Isolation Forest Prediction")
                anomaly = iso.predict(X_input)[0]
                score = -iso.decision_function(X_input)[0]
                st.success(f"**Prediction**: {'Potential Fraud' if anomaly == -1 else 'Normal'}")
                st.write(f"**Anomaly Score**: {score:.3f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.write("Built with Streamlit | Models: Pre-trained XGBoost and Isolation Forest")
st.write("Ensure categorical inputs match training data categories for accurate predictions.")