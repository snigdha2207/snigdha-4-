
# Petite Fashion Analytics Dashboard

A Streamlit dashboard for a petite-first fashion business in India.  
It covers:

- Descriptive analytics
- Diagnostic analytics
- Predictive analytics
  - Classification
  - Clustering
  - Association rule mining
  - Regression
- Prescriptive recommendations
- Future lead scoring through CSV upload

## Files included

- `app.py` — Streamlit app
- `synthetic_petite_fashion_data.csv` — synthetic survey data for 2,000 respondents
- `new_customers_template.csv` — upload template for future leads
- `requirements.txt` — package dependencies

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Upload all files to the **root** of your GitHub repository.
2. Make sure there are **no subfolders required** for the app to run.
3. In Streamlit Community Cloud, connect the repo.
4. Set the main file path as `app.py`.
5. Deploy.

## CSV upload requirements

The uploaded file should contain these columns:

- Age_Group
- Height_Group
- City_Type
- Occupation
- Monthly_Personal_Income
- Body_Shape
- Shopping_Frequency
- Shopping_Channels
- Clothing_Types_Bought
- Fit_Issue_Frequency
- Biggest_Fit_Issues
- Fit_Frustration_Score
- Skipped_Purchase_Due_To_Fit
- Alteration_Frequency
- Monthly_Alteration_Spend
- Online_Return_Frequency
- Return_Reasons
- Preferred_Bottomwear
- Preferred_Topwear
- Preferred_Dress_Types
- Preferred_Colors
- Budget_Per_Item
- Pay_20_Percent_More_For_Perfect_Fit
- Switch_Brand_For_Better_Fit

Use `new_customers_template.csv` as the base format.
