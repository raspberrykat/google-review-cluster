import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Page Conf
st.set_page_config(layout="wide", page_title="Google Reviews Clustering", page_icon="📊")

# Utility Functions (Caching expensive operations)
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Column renaming
    df = df.rename(columns={
        'Category 1': 'churches', 'Category 2': 'resorts', 'Category 3': 'beaches',
        'Category 4': 'parks', 'Category 5': 'theatres', 'Category 6': 'museums',
        'Category 7': 'malls', 'Category 8': 'zoos', 'Category 9': 'restaurants',
        'Category 10': 'pubs/bars', 'Category 11': 'local_services',
        'Category 12': 'burger/pizza_shop', 'Category 13': 'hotels', 'Category 14': 'juice_bars',
        'Category 15': 'art_galleries', 'Category 16': 'dance_clubs', 'Category 17': 'swimming_pools',
        'Category 18': 'gyms', 'Category 19': 'bakeries', 'Category 20': 'beauty_spas',
        'Category 21': 'cafes', 'Category 22': 'view_points', 'Category 23': 'monuments',
        'Category 24': 'gardens', 'Category 25': 'null_column', # Renamed to avoid issues, will likely be dropped
    })
    if 'Unnamed: 25' in df.columns:
        df = df.drop(columns=['Unnamed: 25'])
    return df

@st.cache_data
def preprocess_data(df_raw):
    df = df_raw.copy()
    
    for col in df.columns:
        if col not in ['User']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')

    missing_value_percentage = df.isnull().sum() * 100 / len(df)
    columns_to_drop_missing = []
    for key, value in missing_value_percentage.items():
        if value > 50.0:
            columns_to_drop_missing.append(key)
    if columns_to_drop_missing:
        df = df.drop(columns=columns_to_drop_missing, axis=1)

    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            else: 
                try:
                    mode_val = df[column].mode()
                    if not mode_val.empty:
                        df[column] = df[column].fillna(mode_val[0])
                    else: 
                        df[column] = df[column].fillna("Unknown") 
                except Exception: 
                    df[column] = df[column].fillna("Unknown")

    if 'User' in df.columns:
        df = df.drop(columns=["User"])
    
    df = df.drop_duplicates()
    return df

@st.cache_data
def engineer_features(_df_processed_initial): 
    df_for_eng = _df_processed_initial.copy()
    numeric_cols_orig = df_for_eng.select_dtypes(include=np.number).columns.tolist()
    low_var_features_identified = []
    low_var_df_display_data = []

    if numeric_cols_orig:
        variance = df_for_eng[numeric_cols_orig].var()
        low_var_threshold = 0.2
        low_var_features_identified = variance[variance < low_var_threshold].index.tolist()
        
        low_var_df_display_data = variance[variance < low_var_threshold].reset_index()
        if not low_var_df_display_data.empty: # Check if DataFrame is not empty
            low_var_df_display_data.columns = ['Feature', 'Variance']

        df_after_low_var_drop = df_for_eng.drop(columns=[col for col in low_var_features_identified if col in df_for_eng.columns])
    else:
        df_after_low_var_drop = df_for_eng.copy()

    df_engineered_added = df_after_low_var_drop.copy()
    feature_definitions = {
        'nature': ['beaches', 'parks', 'gardens', 'view_points', 'monuments'],
        'food_and_leisure': ['restaurants', 'cafes', 'bakeries', 'pubs/bars'],
        'health_and_wellness': ['gyms', 'swimming_pools', 'beauty_spas'],
        'art_and_culture': ['art_galleries', 'museums', 'theatres'],
        'retail_and_shop': ['malls', 'local_services', 'burger/pizza_shop'],
        'nightlife': ['dance_clubs', 'pubs/bars']
    }

    all_constituent_cols_used = set()
    for new_feature_name, old_col_names in feature_definitions.items():
        existing_old_cols_in_original = [col for col in old_col_names if col in df_for_eng.columns]
        if existing_old_cols_in_original:
            mean_values = df_for_eng[existing_old_cols_in_original].mean(axis=1, skipna=True)
            df_engineered_added[new_feature_name] = mean_values
            all_constituent_cols_used.update(existing_old_cols_in_original)
        else:
            df_engineered_added[new_feature_name] = np.nan

    constituent_cols_to_drop_final = [col for col in list(all_constituent_cols_used) if col in df_engineered_added.columns]
    df_final_engineered_features = df_engineered_added.drop(columns=constituent_cols_to_drop_final, errors='ignore')
    
    numeric_final_cols = df_final_engineered_features.select_dtypes(include=np.number).columns
    for col in numeric_final_cols:
        if df_final_engineered_features[col].isnull().any():
            df_final_engineered_features[col] = df_final_engineered_features[col].fillna(df_final_engineered_features[col].median())
            
    return df_final_engineered_features, pd.DataFrame(low_var_df_display_data), list(constituent_cols_to_drop_final)


@st.cache_data
def _handle_outliers(df_to_clean): 
    df_cleaned = df_to_clean.copy()
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        for column in numeric_cols:
            q1 = df_cleaned[column].quantile(0.25)
            q3 = df_cleaned[column].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                lower_bound = q1
                upper_bound = q3
            else:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            df_cleaned[column] = df_cleaned[column].clip(lower_bound, upper_bound)
    return df_cleaned

# Streamlit Layout

# Sidebar
st.sidebar.header("🔧 Model Configuration")

REC_SCALER = "QuantileTransformer" 
REC_K = 2 
REC_DIM_RED = "PCA" 

scaler_option_name = st.sidebar.selectbox(
    "Select Scaler",
    ["MinMaxScaler", "StandardScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer", "PowerTransformer", "Normalizer"],
    index=["MinMaxScaler", "StandardScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer", "PowerTransformer", "Normalizer"].index(REC_SCALER)
)

k_option = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2, max_value=10, value=REC_K,
    help=f"Consider Elbow Method/Silhouette Scores for optimal K. Default: {REC_K}." 
)

dim_reduction_option = st.sidebar.selectbox(
    "Select Dimensionality Reduction for Visualization",
    ["PCA", "LDA"],
    index=["PCA", "LDA"].index(REC_DIM_RED)
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Recommendations:**
- **Scaler:** {REC_SCALER}
- **Number of Clusters (K):** {REC_K}
- **Visualization:** {REC_DIM_RED}
""")


# Main Page
st.title("📊 Google Reviews User Clustering")

# Section 1: Problem Statement
st.header("🎯 Problem Statement")
st.markdown("""
This project aims to segment users based on their ratings for various place categories from Google reviews.
By understanding different user segments, businesses or platforms can tailor recommendations, marketing efforts, or services.
The process involves:
1.  Loading and preprocessing user review data.
2.  Performing Exploratory Data Analysis (EDA) to understand data characteristics.
3.  Engineering relevant features from the ratings (with an option for manual feature reduction.)
4.  Applying clustering algorithms (K-Means) to group users.
5.  Visualizing and evaluating the resulting clusters.
""")
st.markdown("---")

# Section 2: Dataset
st.header("💾 Dataset Overview")
df_raw = load_data('google_review_ratings.csv') 
st.subheader("Raw Data Preview (First 5 Rows)")
st.dataframe(df_raw.head())
st.write(f"The raw dataset has **{df_raw.shape[0]}** rows and **{df_raw.shape[1]}** columns.")
st.write(f"Initial column names: `{df_raw.columns.tolist()}`") 
st.markdown("---")

# Section 3: Preprocessing
st.header("⚙️ Data Preprocessing")
with st.expander("Show Preprocessing Details", expanded=False):
    st.markdown("""
    **Steps:**
    1.  **Convert to Numeric:** Ensure all rating-like columns are numeric.
    2.  **Handle Missing Values:**
        - Columns with >50% missing values are dropped.
        - Numeric NaNs are filled with the median.
        - Object NaNs are filled with the mode.
    3.  **Drop 'User' Column:** The 'User' identifier is not needed for clustering features (if present).
    4.  **Remove Duplicates:** Drop duplicate rows.
    """)
df_processed_initial = preprocess_data(df_raw)
st.subheader("Data Preview After Initial Preprocessing (First 5 Rows)")
st.dataframe(df_processed_initial.head())
st.write(f"After initial preprocessing, the dataset has **{df_processed_initial.shape[0]}** rows and **{df_processed_initial.shape[1]}** columns.")
st.write(f"Column names: `{df_processed_initial.columns.tolist()}`")
st.markdown("---")


# Section 4: Feature Engineering
st.header("🛠️ Feature Engineering, Reduction & Outlier Handling")
with st.expander("Show Feature Engineering & Outlier Handling Details", expanded=True): 
    st.markdown("""
    **Feature Engineering Steps:**
    1.  **Drop Low Variance Features:** Original numeric features with variance < 0.2 are removed before aggregation.
    2.  **Create Aggregate Features:** New features are created by averaging ratings for related categories (e.g., 'nature').
    3.  **Drop Constituent Features:** Original features used to create aggregate features are removed.
    4.  **Interactive Feature Dropping:** You can select features to drop based on your analysis (e.g., low correlation, guided by a threshold).
    5.  **Handle Outliers:** Outliers in the final numeric feature set are capped using the IQR method.
    """)

df_engineered_features, low_var_df, dropped_constituents = engineer_features(df_processed_initial)

st.subheader("Low Variance Features (Dropped Before Engineering)")
if not low_var_df.empty:
    st.dataframe(low_var_df)
else:
    st.info("No features met the low variance criteria for dropping from the initial set.")

st.subheader("Constituent Features (Dropped After Engineering)")
if dropped_constituents:
    st.write(f"Constituent features dropped: `{dropped_constituents}`")
else:
    st.info("No constituent features were explicitly dropped.")

st.subheader("Features After Aggregation (Before Interactive Drop & Outlier Handling)")
st.dataframe(df_engineered_features.head())
st.write(f"Dataset after feature aggregation has **{df_engineered_features.shape[0]}** rows and **{df_engineered_features.shape[1]}** columns.")

# Feature Dropping
st.subheader("Interactive Feature Dropping (based on your analysis)")
df_for_interactive_drop = df_engineered_features.copy()
numeric_cols_for_interactive_drop = df_for_interactive_drop.select_dtypes(include=np.number)

features_to_drop_interactive = []
if not numeric_cols_for_interactive_drop.empty and numeric_cols_for_interactive_drop.shape[1] > 1:
    corr_threshold_interactive = st.slider(
        "Set max absolute correlation threshold for drop suggestion:",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Features whose max absolute correlation with any *other single* feature is *less than* this value will be suggested for dropping."
    )
    
    corr_matrix_interactive = numeric_cols_for_interactive_drop.corr()
    
    # Suggest features with max correlation less than the chosen threshold
    # For each feature, find its max correlation with ANY OTHER feature
    max_abs_corr = corr_matrix_interactive.abs().apply(lambda x: sorted(x, reverse=True)[1] if len(x) > 1 else 0, axis=1)
    low_corr_suggestions = max_abs_corr[max_abs_corr < corr_threshold_interactive].index.tolist()
    
    if low_corr_suggestions:
        st.info(f"**Suggestion**: Based on your threshold of **{corr_threshold_interactive}**, features whose maximum absolute correlation with any *other single* feature is less than this value are: `{low_corr_suggestions}`. Consider if these are valuable for clustering.")
    else:
        st.info(f"No features found with maximum absolute correlation to another feature being less than {corr_threshold_interactive} (based on current features).")

    features_to_drop_interactive = st.multiselect(
        "Select features to drop from the current set:",
        options=numeric_cols_for_interactive_drop.columns.tolist(),
        default=low_corr_suggestions if low_corr_suggestions else None,
        help="Chosen features here will be removed before outlier handling and modeling."
    )
    if features_to_drop_interactive:
        df_after_interactive_drop = df_for_interactive_drop.drop(columns=features_to_drop_interactive)
        st.write(f"Features dropped: `{features_to_drop_interactive}`")
    else:
        df_after_interactive_drop = df_for_interactive_drop
else:
    st.write("Not enough numeric features (or only one) for correlation-based suggestions or interactive dropping.")
    df_after_interactive_drop = df_for_interactive_drop


df_numeric_for_outlier = df_after_interactive_drop.select_dtypes(include=np.number)
if df_numeric_for_outlier.empty and not df_after_interactive_drop.empty:
    st.warning("No numeric columns found after feature engineering/dropping to handle outliers.")
    df_cleaned_final_features = df_after_interactive_drop.copy() 
elif df_after_interactive_drop.empty:
    st.error("Feature engineering/dropping resulted in an empty DataFrame. Cannot proceed.")
    if 'stop_app' not in st.session_state: st.session_state.stop_app = True 
    st.stop()
else:
    df_cleaned_final_features = _handle_outliers(df_numeric_for_outlier)
    non_numeric_cols = df_after_interactive_drop.select_dtypes(exclude=np.number)
    if not non_numeric_cols.empty: 
        df_cleaned_final_features = pd.concat([df_cleaned_final_features, non_numeric_cols], axis=1)


st.subheader("Final Features for Modeling (After Interactive Drop & Outlier Handling)")
st.dataframe(df_cleaned_final_features.head())
st.write(f"Final dataset for modeling has **{df_cleaned_final_features.shape[0]}** rows and **{df_cleaned_final_features.shape[1]}** columns.")
st.markdown("---")


# Section 5: Exploratory Data Analysis
st.header("📊 Exploratory Data Analysis")

# EDA: Dataset Selection for Distributions & Basic Stats
st.subheader("Dataset Selection for Distributions & Descriptive Statistics")
eda_dist_dataset_option = st.radio(
    "Choose dataset for visualizing distributions (histograms, boxplots) and descriptive stats (skewness, kurtosis):",
    ("Final Features for Modeling", "Original (Renamed & Cleaned after preprocessing)"),
    index=0, 
    key="eda_dist_dataset_select"
)

df_eda_dist = None 
dataset_name_for_dist_eda = ""

if eda_dist_dataset_option == "Final Features for Modeling":
    if df_cleaned_final_features.empty or df_cleaned_final_features.select_dtypes(include=np.number).empty:
        st.warning("No numeric data in 'Final Features for Modeling'.")
    else:
        df_eda_dist = df_cleaned_final_features.select_dtypes(include=np.number)
        dataset_name_for_dist_eda = "Final Features for Modeling"
elif eda_dist_dataset_option == "Original (Renamed & Cleaned after preprocessing)":
    df_temp_original_numeric = df_processed_initial.select_dtypes(include=np.number) 
    if df_temp_original_numeric.empty:
        st.warning("No numeric data in 'Original (Renamed & Cleaned after preprocessing)'.")
    else:
        df_eda_dist = df_temp_original_numeric
        dataset_name_for_dist_eda = "Original (Renamed & Cleaned after preprocessing) Features"

if df_eda_dist is None or df_eda_dist.empty:
    st.warning(f"No numeric data available for distribution EDA from the selected source: '{dataset_name_for_dist_eda if dataset_name_for_dist_eda else eda_dist_dataset_option}'. Distribution plots and stats will be skipped.")
else:
    st.success(f"Showing distributions and descriptive stats for: **{dataset_name_for_dist_eda}**")

    st.subheader(f"Overall Ratings Distribution ({dataset_name_for_dist_eda})")
    fig_hist_all, ax_hist_all = plt.subplots(figsize=(10, 6))
    melted_df = pd.melt(df_eda_dist)
    if not melted_df['value'].dropna().empty:
        sns.histplot(data=melted_df, x='value', kde=True, ax=ax_hist_all)
        ax_hist_all.set_title(f'Overall Ratings Distribution of {dataset_name_for_dist_eda}')
        ax_hist_all.set_xlabel('Rating Value')
        ax_hist_all.set_ylabel('Count')
    else:
        ax_hist_all.text(0.5, 0.5, "No data to plot for overall distribution.", ha='center', va='center')
    sns.despine()
    st.pyplot(fig_hist_all)
    plt.close(fig_hist_all)

    st.subheader(f"Individual Feature Distributions ({dataset_name_for_dist_eda})")
    eda_col1, eda_col2 = st.columns(2)
    num_cols_eda_list = df_eda_dist.columns.tolist()

    with eda_col1:
        st.write(f"**Histogram for a Selected Feature** (from {dataset_name_for_dist_eda})")
        if num_cols_eda_list:
            feature_to_plot_hist = st.selectbox(
                "Select feature for histogram:",
                num_cols_eda_list,
                index=0 if num_cols_eda_list else None,
                key="hist_select_dist" 
            )
            if feature_to_plot_hist:
                fig_hist_single, ax_hist_single = plt.subplots(figsize=(6, 4))
                sns.histplot(df_eda_dist[feature_to_plot_hist], kde=True, ax=ax_hist_single)
                ax_hist_single.set_title(f'Histogram of {feature_to_plot_hist}')
                st.pyplot(fig_hist_single)
                plt.close(fig_hist_single)
        else:
            st.info("No numeric columns available for individual histograms in the selected dataset.")

    with eda_col2:
        st.write(f"**Boxplot for a Selected Feature** (from {dataset_name_for_dist_eda})")
        if num_cols_eda_list: 
            feature_to_plot_boxplot = st.selectbox(
                "Select feature for boxplot:",
                num_cols_eda_list,
                index=0 if num_cols_eda_list else None,
                key="boxplot_select_dist"
            )
            if feature_to_plot_boxplot:
                fig_boxplot_single, ax_boxplot_single = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df_eda_dist, x=feature_to_plot_boxplot, ax=ax_boxplot_single)
                ax_boxplot_single.set_title(f'Boxplot of {feature_to_plot_boxplot}')
                st.pyplot(fig_boxplot_single)
                plt.close(fig_boxplot_single)
        else:
            st.info("No numeric columns available for boxplots in the selected dataset.")

    st.subheader(f"Overall Boxplot of Numeric Features ({dataset_name_for_dist_eda})")
    fig_boxplot_all, ax_boxplot_all = plt.subplots(figsize=(max(10, df_eda_dist.shape[1]*0.7), 6)) # Dynamic width
    sns.boxplot(data=df_eda_dist, ax=ax_boxplot_all)
    plt.xticks(rotation=45, ha="right")
    ax_boxplot_all.set_title(f"Overall Boxplot of Features in {dataset_name_for_dist_eda}")
    plt.tight_layout()
    st.pyplot(fig_boxplot_all)
    plt.close(fig_boxplot_all)

    st.subheader(f"Skewness and Kurtosis of Numeric Features ({dataset_name_for_dist_eda})")
    skew_kurt_data = []
    for column in df_eda_dist.columns:
        skew_kurt_data.append({
            "Feature": column,
            "Skewness": f"{df_eda_dist[column].skew():.2f}",
            "Kurtosis": f"{df_eda_dist[column].kurtosis():.2f}"
        })
    if skew_kurt_data:
        st.table(pd.DataFrame(skew_kurt_data))
    else: # Should not be reached if df_eda_dist is not empty
        st.info("No numeric features to calculate skewness/kurtosis in the selected dataset.")

st.markdown("---") 

# EDA: Correlation Matrix (always on final features for modeling)
st.subheader("Heatmap: Correlation Matrix of Final Numeric Features (used for modeling)")
df_corr_src = df_cleaned_final_features.select_dtypes(include=np.number) 
if df_corr_src.empty:
     st.info("No numeric features available in the final modeling set to compute its correlation matrix (e.g., all were dropped or data is empty).")
elif df_corr_src.shape[1] <= 1 :
     st.info("Not enough numeric features (need at least 2) in the final modeling set to compute its correlation matrix.")
else:
    corr_matrix_eda = df_corr_src.corr()
    fig_heatmap_eda, ax_heatmap_eda = plt.subplots(figsize=(max(8, df_corr_src.shape[1]*0.8), max(6, df_corr_src.shape[1]*0.6))) 
    sns.heatmap(corr_matrix_eda, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_heatmap_eda)
    ax_heatmap_eda.set_title('Correlation Matrix Heatmap (Features for Modeling)')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig_heatmap_eda)
    plt.close(fig_heatmap_eda)
st.markdown("---")


# Section 6: Model Training & Clustering
st.header("🤖 Model Training & Clustering")

df_for_scaling = df_cleaned_final_features.select_dtypes(include=np.number)

if df_for_scaling.empty:
    st.error("No numeric data available for scaling and clustering. Please check previous steps, especially feature dropping and data processing.")
    if 'stop_app' not in st.session_state: st.session_state.stop_app = True
    st.stop()

if scaler_option_name == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_option_name == "StandardScaler":
    scaler = StandardScaler()
elif scaler_option_name == "RobustScaler":
    scaler = RobustScaler()
elif scaler_option_name == "MaxAbsScaler":
    scaler = MaxAbsScaler()
elif scaler_option_name == "QuantileTransformer":
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
elif scaler_option_name == "PowerTransformer":
    scaler = PowerTransformer(method='yeo-johnson', standardize=True) 
elif scaler_option_name == "Normalizer":
    scaler = Normalizer()

df_scaled_values = scaler.fit_transform(df_for_scaling)
df_scaled = pd.DataFrame(df_scaled_values, columns=df_for_scaling.columns, index=df_for_scaling.index)

st.subheader("Elbow Method for Optimal K")
inertia = []
K_range_elbow = range(1, 11) 
for k_val_elbow in K_range_elbow:
    if k_val_elbow == 0: continue 
    if k_val_elbow > df_scaled.shape[0]: 
        inertia.append(np.nan) 
        continue
    kmeans_elbow = KMeans(n_clusters=k_val_elbow, init='k-means++', random_state=42, n_init='auto')
    kmeans_elbow.fit(df_scaled)
    inertia.append(kmeans_elbow.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
valid_k_elbow = [k for i, k in enumerate(K_range_elbow) if i < len(inertia) and not np.isnan(inertia[i])] # Ensure index is within bounds
valid_inertia = [i for i in inertia if not np.isnan(i)]
if valid_k_elbow and valid_inertia: # Check both are non-empty
    ax_elbow.plot(valid_k_elbow, valid_inertia, 'bo-')
else:
    ax_elbow.text(0.5, 0.5, "Not enough data or K values for Elbow plot", ha='center', va='center')

ax_elbow.set_xlabel('Number of clusters (K)')
ax_elbow.set_ylabel('Inertia')
ax_elbow.set_title('Elbow Method')
st.pyplot(fig_elbow)
plt.close(fig_elbow)

st.subheader(f"K-Means Clustering (K={k_option})")
if k_option > df_scaled.shape[0]:
    st.error(f"Number of selected clusters K={k_option} is greater than the number of samples {df_scaled.shape[0]}. Please choose a smaller K.")
    if 'stop_app' not in st.session_state: st.session_state.stop_app = True
    st.stop()

kmeans = KMeans(n_clusters=k_option, init='k-means++', random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(df_scaled)

df_clustered_display = df_for_scaling.copy() # Use original scaled values for display consistency if needed, or df_for_scaling for original values
df_clustered_display['Cluster'] = cluster_labels

st.write("Cluster Value Counts:")
st.dataframe(df_clustered_display['Cluster'].value_counts().sort_index().reset_index().rename(columns={'Cluster':'Count', 'index':'Cluster'}))


st.subheader(f"Cluster Visualization using {dim_reduction_option}")
columns_for_viz_labels = df_scaled.columns.tolist()
transformed_centers_viz = None
df_dim_reduced = None

if df_scaled.empty or df_scaled.shape[1] == 0:
    st.warning("Scaled data is empty, cannot perform dimensionality reduction for visualization.")
else:
    if dim_reduction_option == "PCA":
        n_pca_components = min(2, df_scaled.shape[1]) 
        if n_pca_components == 0:
            st.error("No features available for PCA.")
            df_dim_reduced = pd.DataFrame()
        else:
            pca = PCA(n_components=n_pca_components, random_state=42)
            df_dim_reduced_vals = pca.fit_transform(df_scaled)
            
            col_names_pca = [f'PCA{i+1}' for i in range(n_pca_components)]
            df_dim_reduced = pd.DataFrame(data=df_dim_reduced_vals, columns=col_names_pca)
            if n_pca_components == 1: 
                df_dim_reduced['PCA2_dummy'] = 0 
            
            if hasattr(kmeans, 'cluster_centers_') and kmeans.cluster_centers_ is not None:
                transformed_centers_viz = pca.transform(kmeans.cluster_centers_)
                if n_pca_components == 1 and transformed_centers_viz.shape[1] == 1:
                    transformed_centers_viz = np.hstack([transformed_centers_viz, np.zeros((transformed_centers_viz.shape[0], 1))])
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_) if n_pca_components > 0 else None

        if df_dim_reduced is not None and not df_dim_reduced.empty:
            fig_pca_cluster, ax_pca_cluster = plt.subplots(figsize=(10, 7))
            y_col_pca = df_dim_reduced.iloc[:,1] if df_dim_reduced.shape[1] > 1 else df_dim_reduced.iloc[:,0] # handle 1D PCA for y
            if df_dim_reduced.shape[1] == 1 or 'PCA2_dummy' in df_dim_reduced.columns: # If effectively 1D
                 y_col_pca = 0 # Plot against a constant y or use a dummy for 1D visualization
            
            sns.scatterplot(x=df_dim_reduced.iloc[:,0], y=y_col_pca, 
                            hue=cluster_labels, palette='viridis', alpha=0.7, ax=ax_pca_cluster, legend='full')
            if transformed_centers_viz is not None:
                ax_pca_cluster.scatter(transformed_centers_viz[:, 0], transformed_centers_viz[:, 1 if transformed_centers_viz.shape[1] > 1 else 0], 
                                       c='red', s=250, marker='o', label='Centroids') 
            ax_pca_cluster.set_title(f'K-Means Clusters (K={k_option}) visualized with PCA')
            ax_pca_cluster.set_xlabel(df_dim_reduced.columns[0])
            ax_pca_cluster.set_ylabel(df_dim_reduced.columns[1] if df_dim_reduced.shape[1] > 1 and 'PCA2_dummy' not in df_dim_reduced.columns[1] else ('Component 2 (or dummy)' if 'PCA2_dummy' in df_dim_reduced.columns else ''))


            if loadings is not None and columns_for_viz_labels:
                for i, feature in enumerate(columns_for_viz_labels):
                    if i < loadings.shape[0]:
                        # Ensure y-component of arrow and text is 0 if only 1 PCA component
                        y_loading = loadings[i, 1] if loadings.shape[1] > 1 and n_pca_components > 1 else 0
                        ax_pca_cluster.arrow(0, 0, loadings[i, 0], y_loading,
                                           color='black', alpha=0.5, head_width=0.02 if n_pca_components > 1 else 0) # No head_width for 1D
                        ax_pca_cluster.text(loadings[i, 0] * 1.15, y_loading * 1.15,
                                           feature, color='black', ha='center', va='center', fontsize=9)
            ax_pca_cluster.legend()
            st.pyplot(fig_pca_cluster)
            plt.close(fig_pca_cluster)

    elif dim_reduction_option == "LDA":
        n_components_lda = min(k_option - 1, df_scaled.shape[1], 2) 
        
        if n_components_lda <= 0:
            st.warning(f"LDA requires at least 1 component and n_classes-1 components. Current K={k_option}, Features={df_scaled.shape[1]}. Cannot visualize with LDA with these settings. Try increasing K or ensuring more features.")
            df_dim_reduced = pd.DataFrame()
        else:
            lda = LDA(n_components=n_components_lda)
            df_dim_reduced_vals = lda.fit_transform(df_scaled, cluster_labels)
            
            col_names_lda = [f'LD{i+1}' for i in range(n_components_lda)]
            df_dim_reduced = pd.DataFrame(data=df_dim_reduced_vals, columns=col_names_lda)
            if n_components_lda == 1: 
                df_dim_reduced['LD2_dummy'] = 0
            
            if hasattr(kmeans, 'cluster_centers_') and kmeans.cluster_centers_ is not None:
                # Need to fit LDA on centroids too, or project them. Projecting is simpler if LDA already fit.
                try:
                    transformed_centers_viz = lda.transform(kmeans.cluster_centers_)
                    if n_components_lda == 1 and transformed_centers_viz.shape[1] == 1:
                        transformed_centers_viz = np.hstack([transformed_centers_viz, np.zeros((transformed_centers_viz.shape[0], 1))])
                except Exception as e:
                    st.warning(f"Could not transform cluster centers for LDA visualization: {e}")
                    transformed_centers_viz = None # Fallback

            scalings = lda.scalings_ if hasattr(lda, 'scalings_') else None

        if df_dim_reduced is not None and not df_dim_reduced.empty:
            fig_lda_cluster, ax_lda_cluster = plt.subplots(figsize=(10, 7))
            sns.scatterplot(x=df_dim_reduced.iloc[:,0], y=df_dim_reduced.iloc[:,1 if df_dim_reduced.shape[1] > 1 else 0], 
                            hue=cluster_labels, palette='viridis', alpha=0.7, ax=ax_lda_cluster, legend='full')
            if transformed_centers_viz is not None:
                 ax_lda_cluster.scatter(transformed_centers_viz[:, 0], transformed_centers_viz[:, 1 if transformed_centers_viz.shape[1] > 1 else 0], 
                                       c='red', s=250, marker='o', label='Centroids') # Changed marker to 'o'
            ax_lda_cluster.set_title(f'K-Means Clusters (K={k_option}) visualized with LDA')
            ax_lda_cluster.set_xlabel(df_dim_reduced.columns[0])
            ax_lda_cluster.set_ylabel(df_dim_reduced.columns[1] if df_dim_reduced.shape[1] > 1 and 'LD2_dummy' not in df_dim_reduced.columns[1] else ('Artificial Y' if 'LD2_dummy' in df_dim_reduced.columns[1] else ''))


            if scalings is not None and columns_for_viz_labels:
                for i, feature in enumerate(columns_for_viz_labels):
                    if i < scalings.shape[0]: 
                        ax_lda_cluster.arrow(0, 0, scalings[i, 0], scalings[i, 1 if scalings.shape[1]>1 and n_components_lda > 1 else 0],
                                           color='black', alpha=0.5, head_width=0.02 if n_components_lda > 1 else 0)
                        ax_lda_cluster.text(scalings[i, 0] * 1.15, scalings[i, 1 if scalings.shape[1]>1 and n_components_lda > 1 else 0] * 1.15,
                                           feature, color='black', ha='center', va='center', fontsize=9)
            ax_lda_cluster.legend()
            st.pyplot(fig_lda_cluster)
            plt.close(fig_lda_cluster)
st.markdown("---")

# Section 7: Model Evaluation
st.header("🏅 Model Evaluation")

if k_option >= 2 and df_scaled.shape[0] > 1 and not df_scaled.empty and len(np.unique(cluster_labels)) > 1:
    if len(np.unique(cluster_labels)) < df_scaled.shape[0] : # Required for silhouette
        sil_score = silhouette_score(df_scaled, cluster_labels)
        db_score = davies_bouldin_score(df_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(df_scaled, cluster_labels)

        st.subheader(f"Metrics for K = {k_option}:")
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Silhouette Score", f"{sil_score:.4f}", help="Higher is better (Range: -1 to 1). Measures how similar an object is to its own cluster compared to other clusters.")
        eval_col2.metric("Davies-Bouldin Index", f"{db_score:.4f}", help="Lower is better (Range: >=0). Measures the average similarity ratio of each cluster with its most similar cluster.")
        eval_col3.metric("Calinski-Harabasz Score", f"{ch_score:.2f}", help="Higher is better. Ratio of the sum of between-cluster dispersion and of inter-cluster dispersion.")

        st.subheader("Metrics Overview for K=2 to 10:")
        metrics_data = []
        K_range_eval = range(2, 11)
        for k_val_eval in K_range_eval:
            if k_val_eval > df_scaled.shape[0] or df_scaled.shape[0] < k_val_eval : 
                continue 
            
            kmeans_eval = KMeans(n_clusters=k_val_eval, init='k-means++', random_state=42, n_init='auto')
            labels_eval = kmeans_eval.fit_predict(df_scaled)
            
            if len(np.unique(labels_eval)) > 1 and len(np.unique(labels_eval)) < df_scaled.shape[0]: 
                metrics_data.append({
                    "K": k_val_eval,
                    "Silhouette": silhouette_score(df_scaled, labels_eval),
                    "Davies-Bouldin": davies_bouldin_score(df_scaled, labels_eval),
                    "Calinski-Harabasz": calinski_harabasz_score(df_scaled, labels_eval)
                })
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data).set_index("K")
            st.dataframe(df_metrics.style.format("{:.3f}"))
        else:
            st.warning("Not enough data points or distinct clusters across the K range to calculate the detailed K metrics table.")
    else:
        st.warning(f"Cannot calculate all clustering evaluation metrics for K={k_option}. Silhouette score, for instance, requires the number of labels to be > 1 and < n_samples. Current unique labels: {len(np.unique(cluster_labels))}, Samples: {df_scaled.shape[0]}.")


else:
    st.warning(f"Cannot calculate clustering evaluation metrics. Ensure K >= 2 (selected K={k_option}), sufficient data points (found {df_scaled.shape[0]}), and more than one cluster formed with enough members (unique labels formed: {len(np.unique(cluster_labels)) if 'cluster_labels' in locals() else 'N/A'}).")

st.markdown("---")
st.subheader("Final Clustered Data Preview (Original Features with Cluster Labels)")
if 'df_clustered_display' in locals() and not df_clustered_display.empty:
    st.dataframe(df_clustered_display.head())
    try:
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df_to_csv(df_clustered_display)
        st.download_button(
            label="📥 Download Full Clustered Data as CSV",
            data=csv,
            file_name='clustered_google_reviews_data.csv',
            mime='text/csv',
            key='download-csv'
    )
    except Exception as e:
        st.error(f"Error {e}")
else:
    st.info("Clustered data preview is not available (likely due to an issue in the clustering step).")

# End of App
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Google Reviews Dataset")

if 'stop_app' in st.session_state and st.session_state.stop_app:
    del st.session_state.stop_app 
    st.stop()
