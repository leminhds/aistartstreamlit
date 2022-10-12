from email.policy import default
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pycountry
import numpy as np
import pickle
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")

df = pd.read_csv('./streamlit_input.csv')
df.dropna(subset="Domain", inplace=True)

countries_columns = list()
for column in df.columns:
        try: 
            pycountry.countries.lookup(column)
            countries_columns.append(column)
        except LookupError:
            pass     

n_founders_df = df.groupby('Domain').size().to_frame('Number of Founders')
df_grouped = df.groupby('Domain').sum()
df_to_print = pd.merge(n_founders_df, df_grouped, left_index=True, right_index=True)



row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('AI Start - Dashboard')
with row0_2:
    st.text("")
    st.subheader('Founder Module Demo')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("""This is a small demo about the analytical capability that we have developed the
     past month. The aim is to keep building further module and use this application as the primary 
     source of MVP for quick feedback gathering. While the visual appealing is important, the main purpose is to test whether the analytic output make sense""")

row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader("General data statistic:")


row2_spacer1, row2_1, row2_spacer2, row2_2   = st.columns((.2, 1.6, .2, 1.6))
with row2_1:
    unique_games_in_df = df['Domain'].nunique()
    str_companies = str(unique_games_in_df) + " Companies"
    st.markdown(str_companies)
with row2_2:
    n_founders = df.shape[0]
    n_founders = str(n_founders) + " Founders"
    st.markdown(n_founders)

filter_selected = st.selectbox('Filter company by criteria:', df_grouped.columns.drop(countries_columns))
row8_1, col_8_2 = st.columns(2)
with row8_1:
    n_year_filtered = st.slider('minimum number of year', 0, 10, value=1, step=1)

try:
    filtered_companies = df_grouped[df_grouped[filter_selected] >=n_year_filtered]


    with open('./modelRF.pkl', 'rb') as f:
        RF_model = pickle.load(f)
    X = filtered_companies.drop('Tracxn Score', axis = 1)
    y = filtered_companies['Tracxn Score']
    explainer = shap.TreeExplainer(RF_model)
    shap_values = explainer.shap_values(X)


    st.markdown('#')
    st.markdown(f'## What Matter most when you analyze a founder while considering {filter_selected} with at least {n_year_filtered} years of experience:')
    fig = plt.figure(figsize=(4, 3))
    shap.summary_plot(shap_values, X, plot_type="bar", )
    st.pyplot(bbox_inches='tight')
    plt.clf()





    row9_1, col_9_2 = st.columns(2)
    with row9_1:
        company_selected = st.multiselect('Select the company you want to analyze', filtered_companies.index)
        all_options = st.checkbox("Select all options")

    with col_9_2:
        columns_selected = st.multiselect("Which are the information that you are interested in?", df_to_print.columns.drop(countries_columns))
        all_options_columns = st.checkbox("Select all columns")

    if all_options_columns:
        columns_to_show = df_to_print.columns.drop(countries_columns)
    else:
        columns_to_show = columns_selected

    if all_options:
        company_selected = df['Domain'].unique()
        st.dataframe(df_to_print.loc[company_selected, columns_to_show])
    else:
        if company_selected:
            st.dataframe(df_to_print.loc[company_selected, columns_to_show])


    columns_to_compare = df_to_print[df_to_print.columns.drop(countries_columns)].columns

    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(data=df['Tracxn Score'], ax=ax)
    if company_selected:
        for company in company_selected:
            plt.annotate(company,
                            xy= (df[df['Domain'] == company]['Tracxn Score'].iloc[0], 10),
                            # Shrink the arrow to avoid occlusion
                            arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03}, alpha=0.3)
        st.pyplot(fig)
        st.markdown('#')
        st.markdown('## Founder Deep Dive')

    row11_1, row11_2 = st.columns(2)
    with row11_1:
        company_to_deep_dive = st.selectbox('Select the company to analyze further about the founder', filtered_companies.index)
    with row11_2:
        columns_selected_deepdive = st.multiselect("Which are the information that you are interested in for the deep dive?", df_to_print.columns.drop(countries_columns))
        all_options_columns = st.checkbox("Select all columns available")

    columsntoshow_deepdive = ['linkedin_url', 'fndr_names']
    if columns_selected_deepdive:
        columsntoshow_deepdive.extend(columns_selected_deepdive)
    if all_options_columns:
        st.dataframe(df[df['Domain'] == company_to_deep_dive][df.columns.drop(countries_columns)])
    else:
        st.dataframe(df[df['Domain'] == company_to_deep_dive][columsntoshow_deepdive])
    # st.dataframe(df[df['Domain'] == company_to_deep_dive])

except ValueError:
    st.markdown(f"No company in our database has such experience. Change your search (try decreasing number of year)")


st.sidebar.selectbox('Select what to analyze', ['Team', 'market', 'traction'])
