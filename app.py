
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Petite Fashion Analytics Dashboard",
    page_icon="👗",
    layout="wide"
)

DATA_FILE = "synthetic_petite_fashion_data.csv"

MULTI_SELECT_COLS = [
    "Shopping_Channels",
    "Clothing_Types_Bought",
    "Biggest_Fit_Issues",
    "Return_Reasons",
    "Preferred_Bottomwear",
    "Preferred_Topwear",
    "Preferred_Dress_Types",
    "Preferred_Colors",
]

CATEGORICAL_COLS = [
    "Age_Group",
    "Height_Group",
    "City_Type",
    "Occupation",
    "Monthly_Personal_Income",
    "Body_Shape",
    "Shopping_Frequency",
    "Fit_Issue_Frequency",
    "Skipped_Purchase_Due_To_Fit",
    "Alteration_Frequency",
    "Monthly_Alteration_Spend",
    "Online_Return_Frequency",
    "Budget_Per_Item",
    "Pay_20_Percent_More_For_Perfect_Fit",
    "Switch_Brand_For_Better_Fit",
]

NUMERIC_COLS = ["Fit_Frustration_Score"]

REQUIRED_INPUT_COLUMNS = [
    "Age_Group","Height_Group","City_Type","Occupation","Monthly_Personal_Income",
    "Body_Shape","Shopping_Frequency","Shopping_Channels","Clothing_Types_Bought",
    "Fit_Issue_Frequency","Biggest_Fit_Issues","Fit_Frustration_Score",
    "Skipped_Purchase_Due_To_Fit","Alteration_Frequency","Monthly_Alteration_Spend",
    "Online_Return_Frequency","Return_Reasons","Preferred_Bottomwear",
    "Preferred_Topwear","Preferred_Dress_Types","Preferred_Colors","Budget_Per_Item",
    "Pay_20_Percent_More_For_Perfect_Fit","Switch_Brand_For_Better_Fit"
]

BUDGET_TO_MIDPOINT = {
    "Below ₹1,000": 750,
    "₹1,000 – ₹2,000": 1500,
    "₹2,000 – ₹3,000": 2500,
    "₹3,000 – ₹5,000": 4000,
    "Above ₹5,000": 6000,
}

INTEREST_MAP = {
    "Definitely will buy": "High",
    "Likely": "High",
    "Maybe": "Medium",
    "Unlikely": "Low",
    "Definitely not": "Low",
}


def split_multi(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


def add_target_columns(df):
    df = df.copy()
    df["Interest_Class_3"] = df["Purchase_Likelihood"].map(INTEREST_MAP)
    df["Interested_Binary"] = np.where(df["Interest_Class_3"] == "High", 1, 0)
    df["Budget_Midpoint"] = df["Budget_Per_Item"].map(BUDGET_TO_MIDPOINT).astype(float)
    return df


def build_feature_matrix(df, fit_mlbs=None):
    base_df = df.copy()

    for col in MULTI_SELECT_COLS:
        base_df[col] = base_df[col].apply(split_multi)

    cat_df = base_df[CATEGORICAL_COLS].copy()
    num_df = base_df[NUMERIC_COLS].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), CATEGORICAL_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), NUMERIC_COLS),
        ],
        remainder="drop"
    )

    X_basic = preprocessor.fit_transform(base_df) if fit_mlbs is None else preprocessor.transform(base_df)
    feature_names = []
    if fit_mlbs is None:
        feature_names.extend(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_COLS).tolist())
        feature_names.extend(NUMERIC_COLS)
    else:
        feature_names.extend(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_COLS).tolist())
        feature_names.extend(NUMERIC_COLS)

    mlb_objects = {} if fit_mlbs is None else fit_mlbs
    extra_arrays = []
    extra_names = []

    for col in MULTI_SELECT_COLS:
        values = base_df[col]
        if fit_mlbs is None:
            mlb = MultiLabelBinarizer()
            arr = mlb.fit_transform(values)
            mlb_objects[col] = mlb
        else:
            mlb = fit_mlbs[col]
            arr = mlb.transform(values)
        names = [f"{col}__{cls}" for cls in mlb.classes_]
        extra_arrays.append(arr)
        extra_names.extend(names)

    if hasattr(X_basic, "toarray"):
        X_basic = X_basic.toarray()

    X = np.hstack([X_basic] + extra_arrays)
    all_feature_names = feature_names + extra_names
    return X, all_feature_names, preprocessor, mlb_objects


def transform_new_data(df_new, preprocessor, mlb_objects):
    working = df_new.copy()
    for col in MULTI_SELECT_COLS:
        working[col] = working[col].apply(split_multi)

    X_basic = preprocessor.transform(working)
    if hasattr(X_basic, "toarray"):
        X_basic = X_basic.toarray()

    extra_arrays = []
    for col in MULTI_SELECT_COLS:
        extra_arrays.append(mlb_objects[col].transform(working[col]))

    X = np.hstack([X_basic] + extra_arrays)
    return X


@st.cache_resource(show_spinner=False)
def train_models():
    df = add_target_columns(load_data())
    X, feature_names, preprocessor, mlb_objects = build_feature_matrix(df)

    y_class = df["Interested_Binary"]
    y_reg = df["Budget_Midpoint"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.25, random_state=42, stratify=y_class
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    full_clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        random_state=42, class_weight="balanced"
    )
    full_clf.fit(X, y_class)

    reg = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42
    )
    reg.fit(X, y_reg)

    cluster_features = [
        "Fit_Frustration_Score", "Budget_Midpoint", "Interested_Binary"
    ]
    cluster_df = df[["Age_Group","Height_Group","City_Type","Occupation","Shopping_Frequency",
                     "Fit_Issue_Frequency","Fit_Frustration_Score","Budget_Midpoint",
                     "Interested_Binary","Online_Return_Frequency","Switch_Brand_For_Better_Fit"]].copy()

    cluster_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Age_Group","Height_Group","City_Type","Occupation","Shopping_Frequency",
                                                             "Fit_Issue_Frequency","Online_Return_Frequency","Switch_Brand_For_Better_Fit"]),
            ("num", StandardScaler(), ["Fit_Frustration_Score","Budget_Midpoint","Interested_Binary"]),
        ]
    )
    X_cluster = cluster_preprocessor.fit_transform(cluster_df)
    if hasattr(X_cluster, "toarray"):
        X_cluster = X_cluster.toarray()

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_cluster)
    df["Cluster"] = cluster_labels

    cluster_summary = df.groupby("Cluster").agg(
        Respondents=("Respondent_ID","count"),
        Avg_Frustration=("Fit_Frustration_Score","mean"),
        Avg_Budget=("Budget_Midpoint","mean"),
        High_Interest_Rate=("Interested_Binary","mean"),
    ).round(2).reset_index()

    # Human-friendly cluster names
    cluster_names = {}
    for _, row in cluster_summary.iterrows():
        c = int(row["Cluster"])
        if row["High_Interest_Rate"] >= 0.75 and row["Avg_Budget"] >= 3200:
            cluster_names[c] = "Premium Fit Seekers"
        elif row["High_Interest_Rate"] >= 0.70:
            cluster_names[c] = "High-Pain Conversion Ready"
        elif row["Avg_Budget"] < 1800:
            cluster_names[c] = "Budget-Conscious Explorers"
        elif row["Avg_Frustration"] < 2.5:
            cluster_names[c] = "Low-Pain Casual Shoppers"
        else:
            cluster_names[c] = "Mainstream Petite Prospects"
    df["Cluster_Name"] = df["Cluster"].map(cluster_names)

    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": full_clf.feature_importances_
    }).sort_values("Importance", ascending=False)

    results = {
        "df": df,
        "clf": full_clf,
        "reg": reg,
        "preprocessor": preprocessor,
        "mlb_objects": mlb_objects,
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "feature_importance": importances,
        "cluster_preprocessor": cluster_preprocessor,
        "kmeans": kmeans,
        "cluster_names": cluster_names,
        "clustered_df": df,
        "cluster_summary": cluster_summary,
    }
    return results


def get_association_results(df):
    basket_cols = ["Preferred_Bottomwear","Preferred_Topwear","Preferred_Dress_Types","Preferred_Colors","Clothing_Types_Bought"]
    combined_items = []
    for _, row in df[basket_cols].iterrows():
        items = []
        for col in basket_cols:
            items.extend(split_multi(row[col]))
        combined_items.append(sorted(set(items)))

    mlb = MultiLabelBinarizer()
    basket = pd.DataFrame(mlb.fit_transform(combined_items), columns=mlb.classes_)
    freq = apriori(basket, min_support=0.07, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(), basket
    rules = association_rules(freq, metric="confidence", min_threshold=0.25)
    if rules.empty:
        return pd.DataFrame(), basket
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules = rules.sort_values(["lift","confidence"], ascending=False)
    return rules, basket


def recommend_action(row):
    if row["Predicted_Interest_Label"] == "High" and row["Predicted_Budget"] >= 3000:
        return "Premium fit-first ads, officewear or elevated styling bundles, low discount."
    if row["Predicted_Interest_Label"] == "High" and row["Predicted_Budget"] < 3000:
        return "Acquisition discount, jeans + top combo, strong fit-pain messaging."
    if row["Predicted_Interest_Label"] == "Medium":
        return "Awareness campaign, testimonials, fit-guide education, limited-time offer."
    return "Low-priority nurture segment; use broad awareness or retarget later."


def cluster_new_customers(df_new, results):
    cluster_df = df_new[["Age_Group","Height_Group","City_Type","Occupation","Shopping_Frequency",
                         "Fit_Issue_Frequency","Fit_Frustration_Score","Budget_Midpoint",
                         "Interested_Binary","Online_Return_Frequency","Switch_Brand_For_Better_Fit"]].copy()
    X_cluster_new = results["cluster_preprocessor"].transform(cluster_df)
    if hasattr(X_cluster_new, "toarray"):
        X_cluster_new = X_cluster_new.toarray()
    labels = results["kmeans"].predict(X_cluster_new)
    names = [results["cluster_names"].get(int(x), f"Cluster {x}") for x in labels]
    return labels, names


def page_overview(results):
    df = results["df"]
    st.title("Petite Fashion Analytics Dashboard")
    st.caption("Descriptive, diagnostic, predictive, and prescriptive analytics for a petite-first fashion business in India.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Respondents", f"{len(df):,}")
    col2.metric("Petite Share", f"{(df['Height_Group'] != 'Above 5’3”').mean()*100:.1f}%")
    col3.metric("High Interest Share", f"{df['Interested_Binary'].mean()*100:.1f}%")
    col4.metric("Avg Budget / Item", f"₹{df['Budget_Midpoint'].mean():,.0f}")

    st.subheader("Founder Summary")
    st.markdown(
        """
        - The best initial target is likely women under **5'3"**, aged **18–35**, shopping at least **monthly**, with **medium to high fit frustration**.
        - Strong pain indicators such as **alteration behavior**, **poor-fit returns**, and **brand-switching willingness** are likely to be key conversion drivers.
        - Product focus should stay on **bottomwear + tops/shirts + dresses**, while using bundles and message tailoring by persona.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Purchase_Likelihood", category_orders={
            "Purchase_Likelihood": ["Definitely will buy","Likely","Maybe","Unlikely","Definitely not"]
        }, title="Purchase Likelihood Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Budget_Per_Item", category_orders={
            "Budget_Per_Item": ["Below ₹1,000","₹1,000 – ₹2,000","₹2,000 – ₹3,000","₹3,000 – ₹5,000","Above ₹5,000"]
        }, title="Budget Per Item Distribution")
        st.plotly_chart(fig, use_container_width=True)


def page_descriptive(results):
    df = results["df"]
    st.header("Descriptive Analytics — What is happening?")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age_Group", title="Age Group Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="City_Type", title="City Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df, x="Fit_Issue_Frequency", title="Fit Issue Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df, x="Occupation", y="Budget_Midpoint", title="Budget by Occupation")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Product Preferences")
    pref_cols = ["Preferred_Bottomwear","Preferred_Topwear","Preferred_Dress_Types","Preferred_Colors"]
    for col in pref_cols:
        exploded = df[col].apply(split_multi).explode().value_counts().reset_index()
        exploded.columns = [col, "Count"]
        fig = px.bar(exploded.head(10), x=col, y="Count", title=f"Top selections: {col.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Table")
    summary = pd.DataFrame({
        "Metric": [
            "Respondents facing fit issues often/always",
            "Respondents who skipped purchase due to fit (yes frequently / sometimes)",
            "Respondents willing to switch brands",
            "Average frustration score",
        ],
        "Value": [
            f"{df['Fit_Issue_Frequency'].isin(['Always','Often']).mean()*100:.1f}%",
            f"{df['Skipped_Purchase_Due_To_Fit'].isin(['Yes, frequently','Yes, sometimes']).mean()*100:.1f}%",
            f"{df['Switch_Brand_For_Better_Fit'].eq('Yes, definitely').mean()*100:.1f}%",
            f"{df['Fit_Frustration_Score'].mean():.2f}",
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_diagnostic(results):
    df = results["df"]
    st.header("Diagnostic Analytics — Why is this happening?")

    c1, c2 = st.columns(2)
    with c1:
        fit_vs_interest = pd.crosstab(df["Fit_Issue_Frequency"], df["Interest_Class_3"], normalize="index").reset_index()
        fig = px.bar(
            fit_vs_interest, x="Fit_Issue_Frequency", y=["High","Medium","Low"],
            title="Interest mix by fit issue frequency", barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        grouped = df.groupby("Switch_Brand_For_Better_Fit", as_index=False)["Interested_Binary"].mean()
        grouped["Interested_Binary"] *= 100
        fig = px.bar(grouped, x="Switch_Brand_For_Better_Fit", y="Interested_Binary",
                     title="High-interest rate by brand-switching intention")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="Interest_Class_3", y="Fit_Frustration_Score",
                     title="Frustration score by interest class")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        rt = pd.crosstab(df["Online_Return_Frequency"], df["Return_Reasons"].str.contains("Poor fit"), normalize="index").reset_index()
        yes_col = True if True in rt.columns else 1
        fig = px.bar(rt, x="Online_Return_Frequency", y=yes_col,
                     title="Share of return reasons involving poor fit")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top conversion drivers")
    top_features = results["feature_importance"].head(15)
    fig = px.bar(top_features.iloc[::-1], x="Importance", y="Feature", orientation="h",
                 title="Feature importance from Random Forest")
    st.plotly_chart(fig, use_container_width=True)


def page_predictive(results):
    st.header("Predictive Analytics — Classification, Clustering, Association Rules, Regression")

    tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Clustering", "Association Rules", "Regression"])

    with tab1:
        metrics = results["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{metrics['precision']:.3f}")
        c3.metric("Recall", f"{metrics['recall']:.3f}")
        c4.metric("F1-score", f"{metrics['f1']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=results["fpr"], y=results["tpr"], mode="lines",
                                         name=f"ROC Curve (AUC={results['roc_auc']:.3f})"))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(roc_fig, use_container_width=True)
        with col2:
            cm = results["confusion_matrix"]
            cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(cm_fig, use_container_width=True)

        st.subheader("Top 20 Feature Importances")
        fi = results["feature_importance"].head(20)
        fig = px.bar(fi.iloc[::-1], x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cluster_summary = results["cluster_summary"].copy()
        cluster_summary["Cluster_Name"] = cluster_summary["Cluster"].map(results["cluster_names"])
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        fig = px.scatter(
            results["clustered_df"],
            x="Fit_Frustration_Score",
            y="Budget_Midpoint",
            color="Cluster_Name",
            hover_data=["Age_Group","Height_Group","Occupation","Purchase_Likelihood"],
            title="Customer personas by frustration and budget"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        rules, basket = get_association_results(results["df"])
        if rules.empty:
            st.warning("No association rules found with the current threshold settings.")
        else:
            st.dataframe(
                rules[["antecedents","consequents","support","confidence","lift"]].head(20),
                use_container_width=True,
                hide_index=True
            )
            fig = px.scatter(
                rules.head(50),
                x="confidence", y="lift", size="support",
                hover_data=["antecedents","consequents"],
                title="Association rules by confidence and lift"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        df = results["df"]
        temp = df.groupby("Interest_Class_3", as_index=False)["Budget_Midpoint"].mean()
        fig = px.bar(temp, x="Interest_Class_3", y="Budget_Midpoint", title="Average predicted spend by interest class")
        st.plotly_chart(fig, use_container_width=True)
        st.info("The regression model estimates budget per item using demographic, behavior, pain, and preference variables.")


def page_prescriptive(results):
    df = results["clustered_df"]
    st.header("Prescriptive Analytics — What should the founder do?")

    high_interest = df[df["Interest_Class_3"] == "High"].copy()

    st.subheader("Priority Customer Segment")
    top_segment = high_interest.groupby(["Height_Group","Age_Group","City_Type"]).size().reset_index(name="Count").sort_values("Count", ascending=False).head(10)
    st.dataframe(top_segment, use_container_width=True, hide_index=True)

    st.subheader("Recommended Launch Strategy")
    st.markdown(
        """
        **Recommended first focus**
        - Women **under 5'3"**, especially in **18–35** age bands.
        - **Tier 1 and Tier 2** shoppers with **medium to high frustration** and **monthly or weekly shopping frequency**.
        - Strong categories: **bottomwear + tops/shirts + dresses**.

        **Recommended marketing playbooks**
        - **Premium Fit Seekers**: push officewear capsules, low discount, fit-first messaging.
        - **Budget-Conscious Explorers**: jeans + top combos, first-purchase offers.
        - **High-Pain Conversion Ready**: testimonials, alteration savings, “no more tailoring” copy.
        - **Low-Pain Casual Shoppers**: nurture with broad awareness rather than heavy acquisition spend.
        """
    )

    st.subheader("Bundle Ideas from Association Signals")
    rules, _ = get_association_results(df)
    if not rules.empty:
        st.dataframe(rules[["antecedents","consequents","confidence","lift"]].head(10), use_container_width=True, hide_index=True)

    st.subheader("Action Grid")
    actions = pd.DataFrame({
        "Segment": ["High budget + high interest", "Low budget + high interest", "Medium interest", "Low interest"],
        "Action": [
            "Premium ads, officewear bundles, limited discounting",
            "Entry price bundles, first-order discount, strong fit benefit message",
            "Education-led campaign, reviews, fit guide, retargeting",
            "Awareness only, low ad spend priority"
        ]
    })
    st.dataframe(actions, use_container_width=True, hide_index=True)


def page_new_customer_scoring(results):
    st.header("Upload New Would-Be Customers")
    st.caption("Upload a CSV file matching the template columns to score future leads and recommend marketing actions.")

    with open("new_customers_template.csv", "rb") as f:
        st.download_button(
            "Download CSV Template",
            data=f.read(),
            file_name="new_customers_template.csv",
            mime="text/csv"
        )

    uploaded = st.file_uploader("Upload new customer CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to score future customers.")
        return

    try:
        df_new = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df_new.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    # Optional target removal
    df_new = df_new[REQUIRED_INPUT_COLUMNS].copy()
    df_new["Budget_Midpoint"] = df_new["Budget_Per_Item"].map(BUDGET_TO_MIDPOINT).fillna(2000)
    df_new["Interested_Binary"] = 0  # placeholder for clustering; overwritten after prediction

    X_new = transform_new_data(df_new, results["preprocessor"], results["mlb_objects"])
    pred_binary = results["clf"].predict(X_new)
    pred_prob = results["clf"].predict_proba(X_new)[:, 1]
    pred_budget = results["reg"].predict(X_new)

    scored = df_new.copy()
    scored["Predicted_Interest_Binary"] = pred_binary
    scored["Interest_Probability"] = np.round(pred_prob, 4)
    scored["Predicted_Interest_Label"] = np.where(scored["Predicted_Interest_Binary"] == 1, "High", "Low/Medium")
    scored["Predicted_Budget"] = np.round(pred_budget, 0).astype(int)
    scored["Interested_Binary"] = scored["Predicted_Interest_Binary"]

    cluster_ids, cluster_names = cluster_new_customers(scored, results)
    scored["Predicted_Cluster"] = cluster_ids
    scored["Predicted_Cluster_Name"] = cluster_names
    scored["Recommended_Marketing_Action"] = scored.apply(recommend_action, axis=1)

    st.subheader("Scored customers")
    st.dataframe(scored, use_container_width=True)

    csv_buffer = io.StringIO()
    scored.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Predictions",
        data=csv_buffer.getvalue(),
        file_name="scored_new_customers.csv",
        mime="text/csv"
    )


def main():
    results = train_models()
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Descriptive Analytics",
            "Diagnostic Analytics",
            "Predictive Analytics",
            "Prescriptive Analytics",
            "Upload New Customers"
        ]
    )

    if page == "Overview":
        page_overview(results)
    elif page == "Descriptive Analytics":
        page_descriptive(results)
    elif page == "Diagnostic Analytics":
        page_diagnostic(results)
    elif page == "Predictive Analytics":
        page_predictive(results)
    elif page == "Prescriptive Analytics":
        page_prescriptive(results)
    else:
        page_new_customer_scoring(results)

    st.sidebar.markdown("---")
    st.sidebar.write("Model stack")
    st.sidebar.write("- Random Forest Classifier")
    st.sidebar.write("- KMeans Clustering")
    st.sidebar.write("- Apriori Association Rules")
    st.sidebar.write("- Random Forest Regressor")


if __name__ == "__main__":
    main()
