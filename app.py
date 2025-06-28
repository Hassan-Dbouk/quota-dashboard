import streamlit as st
from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
from google.oauth2 import service_account
import json

# --- Streamlit Config ---
st.set_page_config(layout="wide")#


st.markdown("""
    <style>
    /* Give more breathing space at top */
    .main .block-container {
        padding-top: 1.2rem !important;
    }

    /* h1 title styling */
    h1 {
        font-size: 22px !important;
        line-height: 1.2 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Shrink subheadings */
    h2, h3, h4, h5, h6 {
        font-size: 16px !important;
        margin-bottom: 0.3rem !important;
    }

    /* Control font size of other elements */
    .css-10trblm, .css-1v0mbdj p {
        font-size: 13px !important;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)







st.markdown("""
    <style>
    h3 {
        margin-bottom: 0.2rem !important;
    }

    h4 {
        margin-top: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)














# --- Auth Setup ---
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = bigquery.Client(credentials=credentials, project=credentials.project_id)


# --- Loaders ---
@st.cache_data(ttl=3600)
def load_applications():
    query = """
    SELECT
        SAFE.PARSE_DATE('%Y-%m-%d', SUBSTR(`Application Created`, 1, 10)) AS application_date,
        `Nationality Category Updated` AS nationality,
        `Nationality Updated` AS nationality_raw,
        `Location Category Updated` AS location,
        `Activa Visa Status` AS active_visa_status
    FROM `data-driven-attributes.AT_marketing_db.ATD_New_Last_Action_by_User_PivotData_View`
    WHERE `Application Created` IS NOT NULL
    """
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_quotas():
    query = """
    SELECT
        `Nationality Category Updated` AS nationality,
        `Location Category Updated` AS location,
        `Daily Quota Regardless Active Visas` AS quota_all,
        `Daily Quota Considering Active Visas` AS quota_active
    FROM `data-driven-attributes.AT_marketing_db.ATD_Daily_Quotas`
    """
    return client.query(query).to_dataframe()

@st.cache_data(ttl=3600)
def load_spend():
    query = """
    SELECT
      application_created_date AS Day,
      nationality_category AS Nationality,
      location_category AS Location,
      SUM(total_spend_aed) AS Spend
    FROM `data-driven-attributes.AT_marketing_db.AT_Country_Daily_Performance_Spend_ERP_Updated`
    GROUP BY Day, Nationality, Location
    """
    return client.query(query).to_dataframe()


# --- Filters ---
def filter_applications(df, nat, loc, active_only):
    df = df[(df["nationality"] == nat) & (df["location"] == loc)]
    if active_only and nat.lower() == "filipina" and loc.lower() == "philippines":
        df = df[df["active_visa_status"] == "true"]
    return df

def get_daily_quota(df, nat, loc, active_only):
    row = df[(df["nationality"] == nat) & (df["location"] == loc)]
    if row.empty:
        return 0
    return row["quota_active"].iloc[0] if active_only else row["quota_all"].iloc[0]

# --- Quarter Filter ---
today = pd.to_datetime("today").normalize()
this_year = today.year
last_year = this_year - 1

quarter = (today.month - 1) // 3 + 1
quarter_start = datetime(this_year, 3 * (quarter - 1) + 1, 1)
quarter_end = datetime(this_year, 3 * quarter, 1) + pd.offsets.MonthEnd(0)
quarter_months = [quarter_start.month + i for i in range(3)]

# --- Plotting Functions ---
def prepare_grouped(df, level, this_year, last_year, today):
    df = df.copy()
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["year"] = df["application_date"].dt.year
    df = df[df["year"].isin([this_year, last_year])]

    if level == "M":
        df = df[df["application_date"].dt.month.isin(quarter_months)]
        df["period"] = df["application_date"].dt.strftime('%b')
        period_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df["period"] = pd.Categorical(df["period"], categories=period_order, ordered=True)

    elif level == "W":
        df["period"] = df["application_date"].dt.isocalendar().week.astype(int)        
    elif level == "D":
        df["period"] = df["application_date"].dt.strftime('%d/%m')

    grouped = df.groupby(["period", "year"]).size().reset_index(name="applications")
    grouped["applications"] = grouped["applications"].clip(lower=0)
    pivoted = grouped.pivot(index="period", columns="year", values="applications")

    if level == "D" and this_year in pivoted.columns:
        df_cutoff = df[(df["year"] == this_year) & (df["application_date"] <= today)]
        valid_days = df_cutoff["period"].unique()
        invalid_days = [day for day in pivoted.index if day not in valid_days]
        pivoted.loc[invalid_days, this_year] = float('nan')

    if level in ["W", "D"]:
        pivoted = pivoted.sort_index(
            key=lambda x: [int(i.split('/')[1]) * 100 + int(i.split('/')[0]) if isinstance(i, str) else int(i) for i in x]
        )

    return pivoted

def plot_chart(df, title, this_year, last_year, needed_avg, level, today):
    needed_avg = max(0, needed_avg)
    fig = go.Figure()
    if last_year in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[last_year].clip(lower=0), name=f"{last_year}", line=dict(color='gray', dash='dot'), mode='lines'))
    if this_year in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[this_year].clip(lower=0), name=f"{this_year}", line=dict(color='green'), mode='lines'))
        future_x = [x for x in df.index if pd.isna(df.loc[x, this_year])]
        if future_x:
            fig.add_trace(go.Scatter(x=future_x, y=[needed_avg]*len(future_x), name="Required Avg", line=dict(color='red', dash='dot'), mode='lines'))

    fig.update_layout(title=title, xaxis_title="", yaxis_title="Applications", hovermode="x unified", showlegend=True)
    
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickformat=",.0f")
    if level == "D":
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=df.index[::7],
                ticktext=list(df.index[::7])
            )
        )

    st.plotly_chart(fig, use_container_width=True)

# --- Trend Calculations ---
def calculate_wow(df):
    df = df[df["application_date"].notna()]
    end_period = today - timedelta(days=1)
    start_period = end_period - timedelta(days=6)
    prev_start = start_period - timedelta(days=7)
    prev_end = start_period - timedelta(days=1)
    curr_count = df[(df["application_date"] >= start_period) & (df["application_date"] <= end_period)].shape[0]
    prev_count = df[(df["application_date"] >= prev_start) & (df["application_date"] <= prev_end)].shape[0]
    if prev_count == 0:
        return float('nan')
    return ((curr_count - prev_count) / prev_count) * 100

def calculate_mom(df):
    df = df[df["application_date"].notna()]
    days_so_far = today.day
    this_month_start = today.replace(day=1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    this_month_range = (this_month_start, this_month_start + timedelta(days=days_so_far - 1))
    last_month_range = (last_month_start, last_month_start + timedelta(days=days_so_far - 1))
    curr_count = df[(df["application_date"] >= this_month_range[0]) & (df["application_date"] <= this_month_range[1])].shape[0]
    prev_count = df[(df["application_date"] >= last_month_range[0]) & (df["application_date"] <= last_month_range[1])].shape[0]
    if prev_count == 0:
        return float('nan')
    return ((curr_count - prev_count) / prev_count) * 100

def calculate_yoy(df):
    df = df[df["application_date"].notna()]
    this_q = df[(df["application_date"] >= quarter_start) & (df["application_date"] <= today)]
    same_start_last = quarter_start.replace(year=last_year)
    same_end_last = today.replace(year=last_year)
    last_q = df[(df["application_date"] >= same_start_last) & (df["application_date"] <= same_end_last)]
    curr = this_q.shape[0]
    prev = last_q.shape[0]
    if prev == 0:
        return float('nan')
    return ((curr - prev) / prev) * 100

# --- Chart Section ---
st.title(f"MaidsAT Tracker – Q{quarter} Overview")
view_option = st.selectbox("Select Granularity", options=["Daily", "Weekly"], index=0)
granularity_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}
selected_level = granularity_map[view_option]

# --- Load Data ---
df_apps = load_applications()
df_quotas = load_quotas()
df_spend = load_spend()


col1, col2 = st.columns(2)

with col1:
    df_outside = filter_applications(df_apps, "filipina", "outside_uae", active_only=False)
    df_outside["application_date"] = pd.to_datetime(df_outside["application_date"], errors="coerce")
    df_outside_q = df_outside[df_outside["application_date"].dt.month.isin(quarter_months)]

    daily_quota_out = get_daily_quota(df_quotas, "filipina", "outside_uae", active_only=False)
    total_quota_out = daily_quota_out * ((quarter_end - quarter_start).days + 1)
    attained_out = df_outside_q[(df_outside_q["application_date"].dt.year == this_year)].shape[0]
    remaining_days_out = (quarter_end - today).days
    needed_avg_out = max(0, (total_quota_out - attained_out) / remaining_days_out) if remaining_days_out > 0 else 0
    days_so_far_out = (today - quarter_start).days
    avg_so_far_out = attained_out / days_so_far_out if days_so_far_out > 0 else 0
    forecast_out = attained_out + (avg_so_far_out * remaining_days_out)

    wow_out = calculate_wow(df_outside)
    mom_out = calculate_mom(df_outside)
    yoy_out = calculate_yoy(df_outside)

    df_inside = filter_applications(df_apps, "filipina", "inside_uae", active_only=False)
    df_inside["application_date"] = pd.to_datetime(df_inside["application_date"], errors="coerce")
    df_inside_q = df_inside[df_inside["application_date"].dt.month.isin(quarter_months)]

    daily_quota_in = get_daily_quota(df_quotas, "filipina", "inside_uae", active_only=False)
    total_quota_in = daily_quota_in * ((quarter_end - quarter_start).days + 1)
    attained_in = df_inside_q[(df_inside_q["application_date"].dt.year == this_year)].shape[0]
    remaining_days_in = (quarter_end - today).days
    needed_avg_in = max(0, (total_quota_in - attained_in) / remaining_days_in) if remaining_days_in > 0 else 0
    days_so_far_in = (today - quarter_start).days
    avg_so_far_in = attained_in / days_so_far_in if days_so_far_in > 0 else 0
    forecast_in = attained_in + (avg_so_far_in * remaining_days_in)

    wow_in = calculate_wow(df_inside)
    mom_in = calculate_mom(df_inside)
    yoy_in = calculate_yoy(df_inside)

    df_plot_out = prepare_grouped(df_outside_q, selected_level, this_year, last_year, today)
    plot_chart(df_plot_out, "Filipina - Outside UAE", this_year, last_year, needed_avg_out, selected_level, today)

    # st.markdown("#### Filipinas Inside & Outside UAE Summary")    

    # --- Define date range for spend filtering ---
    start_date = quarter_start.date()
    end_date = quarter_end.date()

    # OUTSIDE UAE FILIPINA
    spend_out = df_spend[
        (df_spend["Nationality"] == "filipina") &
        (df_spend["Location"] == "outside_uae") &
        (df_spend["Day"] >= start_date) &
        (df_spend["Day"] <= end_date)
    ]["Spend"].sum()

    cac_out = spend_out / attained_out if attained_out else 0


    # --- INSIDE UAE CAC ---
    spend_in = df_spend[
        (df_spend["Nationality"] == "filipina") &
        (df_spend["Location"] == "inside_uae") &
        (df_spend["Day"] >= start_date) &
        (df_spend["Day"] <= end_date)
    ]["Spend"].sum()

    cac_in = spend_in / attained_in if attained_in else 0


    table_df = pd.DataFrame({
        "Segment": ["OUTAE Apps", "INUAE Apps"],
        "Quota": [total_quota_out, total_quota_in],
        "Delivered": [attained_out, attained_in],
        "%D": [attained_out / total_quota_out * 100, attained_in / total_quota_in * 100],
        "CAC": [cac_out, cac_in],
        "Forecast": [forecast_out, forecast_in],
        "%F": [forecast_out / total_quota_out * 100, forecast_in / total_quota_in * 100],
        "WoW": [wow_out, wow_in],
        "MoM": [mom_out, mom_in],
        "YoY": [yoy_out, yoy_in]
    })

    # --- Format table manually ---
    formatted_table_df = table_df.copy()
    percent_cols = ["%D", "%F", "WoW", "MoM", "YoY"]

    # Format integer columns with comma separator
    for col in ["Quota", "Delivered", "Forecast", "CAC"]:
        formatted_table_df[col] = formatted_table_df[col].apply(lambda x: f"{x:,.0f}")

    # Format percent columns with % and no decimals
    for col in percent_cols:
        formatted_table_df[col] = formatted_table_df[col].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")

    # --- Display table with HTML styling ---
    st.markdown("#### Filipinas Inside & Outside UAE Summary")

    st.markdown(
        formatted_table_df.to_html(
            index=False,
            justify="center",
            border=0,
            classes="centered-table"
        ),
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
        .centered-table {
            width: 100%;
            table-layout: fixed;
            font-size: 13px;
        }
        .centered-table th, .centered-table td {
            text-align: center !important;
            padding: 6px;
            white-space: nowrap;
        }
        /* 👇 Target the first column (Segment) */
        .centered-table td:first-child {
            font-size: 11px;
        }
        </style>
    """, unsafe_allow_html=True)


with col2:
    df_phil = filter_applications(df_apps, "filipina", "philippines", active_only=True)
    df_phil["application_date"] = pd.to_datetime(df_phil["application_date"], errors="coerce")
    df_phil_q = df_phil[df_phil["application_date"].dt.month.isin(quarter_months)]

    daily_quota_ph = get_daily_quota(df_quotas, "filipina", "philippines", active_only=True)
    total_quota_ph = daily_quota_ph * ((quarter_end - quarter_start).days + 1)
    attained_ph = df_phil_q[(df_phil_q["application_date"].dt.year == this_year)].shape[0]
    remaining_days_ph = (quarter_end - today).days
    needed_avg_ph = max(0, (total_quota_ph - attained_ph) / remaining_days_ph) if remaining_days_ph > 0 else 0

    df_plot_ph = prepare_grouped(df_phil_q, selected_level, this_year, last_year, today)
    plot_chart(df_plot_ph, "Filipina - Philippines (Active Visas)", this_year, last_year, needed_avg_ph, selected_level, today)

    st.markdown("#### Philippines Summary")
    start_date = quarter_start.date()
    end_date = quarter_end.date()

    # --- ROW 1: Applications ---
    # All Filipina apps in PH (regardless of visa), cohorted to this quarter
    df_phil_all = filter_applications(df_apps, "filipina", "philippines", active_only=False)
    df_phil_all["application_date"] = pd.to_datetime(df_phil_all["application_date"], errors="coerce")
    df_phil_all_q = df_phil_all[df_phil_all["application_date"].dt.month.isin(quarter_months)]
    apps_ph = df_phil_all_q[df_phil_all_q["application_date"].dt.year == this_year].shape[0]



    daily_quota_apps = get_daily_quota(df_quotas, "filipina", "philippines", active_only=False)
    total_quota_apps = daily_quota_apps * ((quarter_end - quarter_start).days + 1)
    days_so_far_apps = (today - quarter_start).days
    remaining_days_apps = (quarter_end - today).days
    avg_so_far_apps = apps_ph / days_so_far_apps if days_so_far_apps > 0 else 0
    forecast_apps = apps_ph + (avg_so_far_apps * remaining_days_apps)

    # --- ROW 2: Visas ---
    # Same dataset but only where active_visa_status = true
    df_phil_visas_q = df_phil_all_q[df_phil_all_q["active_visa_status"] == "true"]
    visas_ph = df_phil_visas_q[df_phil_visas_q["application_date"].dt.year == this_year].shape[0]

    daily_quota_visas = get_daily_quota(df_quotas, "filipina", "philippines", active_only=True)
    total_quota_visas = daily_quota_visas * ((quarter_end - quarter_start).days + 1)
    days_so_far_visas = (today - quarter_start).days
    remaining_days_visas = (quarter_end - today).days
    avg_so_far_visas = visas_ph / days_so_far_visas if days_so_far_visas > 0 else 0
    forecast_visas = visas_ph + (avg_so_far_visas * remaining_days_visas)


    spend_ph = df_spend[
        (df_spend["Nationality"] == "filipina") &
        (df_spend["Location"] == "philippines") &
        (df_spend["Day"] >= start_date) &
        (df_spend["Day"] <= end_date)
    ]["Spend"].sum()

    cac_apps = spend_ph / apps_ph if apps_ph else 0
    cac_visas = spend_ph / visas_ph if visas_ph else 0

    # --- Table Build ---
    philippines_table = pd.DataFrame({
        "Segment": ["Apps", "Visas"],
        "Quota": [total_quota_apps, total_quota_visas],
        "Delivered": [apps_ph, visas_ph],
        "%D": [
            apps_ph / total_quota_apps * 100 if total_quota_apps else 0,
            visas_ph / total_quota_visas * 100 if total_quota_visas else 0
        ],
        "CAC": [cac_apps, cac_visas],
        "Forecast": [forecast_apps, forecast_visas],
        "%F": [
            forecast_apps / total_quota_apps * 100 if total_quota_apps else 0,
            forecast_visas / total_quota_visas * 100 if total_quota_visas else 0
        ],
        "WoW": [calculate_wow(df_phil_all_q), calculate_wow(df_phil_visas_q)],
        "MoM": [calculate_mom(df_phil_all_q), calculate_mom(df_phil_visas_q)],
        "YoY": [calculate_yoy(df_phil_all_q), calculate_yoy(df_phil_visas_q)],
    })


# --- Format table manually ---
    formatted_philippines_table = philippines_table.copy()
    percent_cols = ["%D", "%F", "WoW", "MoM", "YoY"]

    for col in ["Quota", "Delivered", "Forecast", "CAC"]:
        formatted_philippines_table[col] = formatted_philippines_table[col].apply(lambda x: f"{x:,.0f}")
    for col in percent_cols:
        formatted_philippines_table[col] = formatted_philippines_table[col].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")

    st.markdown(
        formatted_philippines_table.to_html(
            index=False,
            justify="center",
            border=0,
            classes="centered-table"
        ),
        unsafe_allow_html=True
    )


def compute_metrics(df, quotas, nationality_value, nationality_filter, location, label):
    df_group = df[nationality_filter & (df['location'] == location)]
    df_group["application_date"] = pd.to_datetime(df_group["application_date"], errors="coerce")
    df_group_q = df_group[df_group["application_date"].dt.month.isin(quarter_months)]

    # Use the nationality_value explicitly passed in
    daily_quota = get_daily_quota(quotas, nationality_value, location, active_only=False)
    total_quota = daily_quota * ((quarter_end - quarter_start).days + 1)
    attained = df_group_q[df_group_q["application_date"].dt.year == this_year].shape[0]
    remaining_days = (quarter_end - today).days
    needed_avg = max(0, (total_quota - attained) / remaining_days) if remaining_days > 0 else 0
    days_so_far = (today - quarter_start).days
    avg_so_far = attained / days_so_far if days_so_far > 0 else 0
    forecast = attained + (avg_so_far * remaining_days)

    wow = calculate_wow(df_group)
    mom = calculate_mom(df_group)
    yoy = calculate_yoy(df_group)

    # Get CAC
    spend = df_spend[
        (df_spend["Nationality"].str.lower() == nationality_value) &
        (df_spend["Location"] == location) &
        (df_spend["Day"] >= start_date) &
        (df_spend["Day"] <= end_date)
    ]["Spend"].sum()

    cac = spend / attained if attained else 0


    return {
        "Segment": label,
        "Quota": total_quota,
        "Delivered": attained,
        "%D": attained / total_quota * 100 if total_quota else 0,
        "CAC": cac,
        "Forecast": forecast,
        "%F": forecast / total_quota * 100 if total_quota else 0,
        "WoW": wow,
        "MoM": mom,
        "YoY": yoy,
    }





# Your code is correct, but it breaks because of the column name mismatches.
# Here's the fixed version you should paste at the end of your current script (or use this updated script instead):

segments = [
    compute_metrics(
        df_apps,
        df_quotas,
        "african",
        (df_apps["nationality_raw"].str.lower() != "ethiopian") & (df_apps["nationality"].str.lower() == "african"),
        "outside_uae",
        "Africans Apps",
    ),
    compute_metrics(
        df_apps,
        df_quotas,
        "ethiopian",
        (df_apps["nationality_raw"].str.lower() == "ethiopian") & (df_apps["nationality"].str.lower() == "african"),
        "outside_uae",
        "Ethiopians Apps",
    ),
]

# Show the table
st.markdown("#### Africans / Ethiopians Application Summary")
df_afro = pd.DataFrame(segments)


formatted_afro = df_afro.copy()
percent_cols = ["%D", "%F", "WoW", "MoM", "YoY"]

for col in ["Quota", "Delivered", "Forecast", "CAC"]:
    formatted_afro[col] = formatted_afro[col].apply(lambda x: f"{x:,.0f}")
# Format CAC with 1 decimal
formatted_afro["CAC"] = df_afro["CAC"].apply(lambda x: f"{x:,.1f}")
for col in percent_cols:
    formatted_afro[col] = formatted_afro[col].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")




st.markdown(
    formatted_afro.to_html(
        index=False,
        justify="center",
        border=0,
        classes="centered-table"
    ),
    unsafe_allow_html=True
)


st.markdown("""
    <style>
    .centered-table {
        width: 100%;
        table-layout: fixed;
        font-size: 13px;
    }
    .centered-table th, .centered-table td {
        text-align: center !important;
        padding: 6px;
        white-space: nowrap;
    }
    .centered-table td:first-child {
        font-size: 11px;
    }
    </style>
""", unsafe_allow_html=True)