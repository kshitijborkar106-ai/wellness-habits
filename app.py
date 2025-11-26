# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Wellness Habit Analysis", page_icon="🫶", layout="wide")
st.title("🫶 Daily Lifestyle & Wellness Habit Analysis (No-ML)")
st.caption("Upload your Daily_Habit_Tracker.csv to get visual insights, correlation matrix and recommendations. (No DB, No ML)")

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def parse_features(df):
    def t2m(x):
        try:
            h,m = str(x).split(":"); return int(h)*60 + int(m)
        except:
            return np.nan
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Wake_Up_Time' in df.columns:
        df['Wake_Minutes'] = df['Wake_Up_Time'].apply(t2m)
        df['Wake_Hour'] = (df['Wake_Minutes']/60).round(2)
    df['Weekday'] = df['Date'].dt.day_name()
    if 'Water_Intake_ml' in df.columns:
        df['Water_Liters'] = df['Water_Intake_ml'] / 1000.0
    return df

# rule-based recommendations
def recommend_for_values(sleep, steps, water_ml, study, wake_hour, mood=None):
    tips = []
    if pd.notna(sleep):
        if sleep < 6:
            tips.append("😴 Sleep seems low → aim for 7–8 hours nightly.")
        elif sleep > 9.5:
            tips.append("🛌 Very long sleep — aim for consistent 7–9 hrs.")
        else:
            tips.append("✅ Sleep within healthy range — keep it regular.")
    if pd.notna(steps):
        if steps < 5000:
            tips.append("🚶 Steps low → add short walks; target 8k+ steps/day.")
        elif steps < 8000:
            tips.append("🏃 Almost there — add a 15-min brisk walk.")
        else:
            tips.append("✅ Activity level good — maintain variety.")
    if pd.notna(water_ml):
        if water_ml < 2000:
            tips.append("💧 Hydration low → aim for 2–3 L/day.")
        elif water_ml > 4000:
            tips.append("💧 Very high — ensure balance.")
        else:
            tips.append("✅ Hydration seems adequate.")
    if pd.notna(study):
        if study > 8:
            tips.append("🧠 Long study hours — take breaks every hour.")
        elif study < 2:
            tips.append("🗓 Try Pomodoro (25/5) sessions.")
        else:
            tips.append("✅ Study routine balanced.")
    if pd.notna(wake_hour):
        if wake_hour > 10:
            tips.append("⏰ Late wake-up — shift earlier for better circadian rhythm.")
        elif wake_hour < 4.5:
            tips.append("⏰ Very early wake — ensure adequate sleep.")
        else:
            tips.append("✅ Wake-up time reasonable.")
    if mood is not None and pd.notna(mood):
        if mood < 5:
            tips.append("🌤 Low mood — sunlight and short walks can help.")
        elif mood >= 8:
            tips.append("🥳 Good mood — note and repeat what helps.")
    return tips

def dataset_recommendations(df):
    s = df.get('Sleep_Hours').mean()
    steps = df.get('Steps').mean()
    water = df.get('Water_Intake_ml').mean()
    study = df.get('Study_Hours').mean()
    wake = df.get('Wake_Hour').mean()
    mood = df.get('Mood_Score').mean()
    recs = recommend_for_values(s, steps, water, study, wake, mood)
    numeric = df.select_dtypes(include=['float64','int64'])
    if {'Sleep_Hours','Mood_Score'} <= set(numeric.columns):
        c = numeric['Sleep_Hours'].corr(numeric['Mood_Score'])
        if pd.notna(c) and abs(c) >= 0.25:
            recs.append("📈 Sleep and Mood show correlation.")
    if {'Steps','Mood_Score'} <= set(numeric.columns):
        c = numeric['Steps'].corr(numeric['Mood_Score'])
        if pd.notna(c) and c >= 0.25:
            recs.append("🏅 Higher activity correlates with better mood.")
    return recs

# Sidebar
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload Daily_Habit_Tracker.csv", type=["csv"])
if st.sidebar.button("Download sample CSV"):
    sample_df = pd.DataFrame({
        "Date": pd.date_range(end=pd.Timestamp.today(), periods=7).strftime("%Y-%m-%d"),
        "Wake_Up_Time": ["06:30","07:00","06:45","07:15","06:50","07:10","07:00"],
        "Sleep_Hours": [7.0,6.5,7.5,6.0,8.0,7.2,6.8],
        "Steps": [5000,4200,8000,3000,10000,7200,6500],
        "Calories_Burned": [2100,2000,2300,1800,2600,2200,2050],
        "Water_Intake_ml": [2000,1500,2500,1800,3000,2200,2000],
        "Study_Hours": [2,3,1.5,4,2.5,3,2],
        "Mood_Score": [7,6,8,5,9,7,6]
    })
    st.sidebar.download_button("Download sample CSV", sample_df.to_csv(index=False), file_name="sample_daily_habit.csv", mime="text/csv")

df = None
if uploaded:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","📈 Visualizations","🔎 Insights","📝 Recommendations"])

with tab1:
    if df is None:
        st.info("Upload CSV to view dashboard.")
    else:
        df = parse_features(df)
        st.subheader("Quick KPIs")
        c1,c2,c3,c4 = st.columns(4)
        if 'Sleep_Hours' in df: c1.metric("Avg Sleep (hrs)", f"{df['Sleep_Hours'].mean():.2f}")
        if 'Steps' in df: c2.metric("Avg Steps", f"{int(df['Steps'].mean()):,}")
        if 'Water_Intake_ml' in df: c3.metric("Avg Water (L)", f"{df['Water_Intake_ml'].mean()/1000:.2f}")
        if 'Mood_Score' in df: c4.metric("Avg Mood (1–10)", f"{df['Mood_Score'].mean():.2f}")
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

with tab2:
    if df is None:
        st.info("Upload CSV to view charts.")
    else:
        df = parse_features(df)
        st.subheader("Charts")
        if {'Date','Steps'} <= set(df.columns):
            st.markdown("**Steps over time**")
            st.line_chart(df.sort_values('Date').set_index('Date')['Steps'])
        if {'Weekday','Sleep_Hours'} <= set(df.columns):
            st.markdown("**Average Sleep by Weekday**")
            wk = df.groupby('Weekday')['Sleep_Hours'].mean().reindex(
                ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            st.bar_chart(wk)
        if {'Sleep_Hours','Mood_Score'} <= set(df.columns):
            st.markdown("**Sleep vs Mood**")
            st.scatter_chart(df[['Sleep_Hours','Mood_Score']])
        if 'Water_Intake_ml' in df.columns:
            st.markdown("**Water Intake distribution**")
            st.bar_chart(df['Water_Intake_ml'].value_counts().sort_index())

with tab3:
    if df is None:
        st.info("Upload CSV to view insights.")
    else:
        df = parse_features(df)
        st.subheader("Auto Insights")
        insights = []
        for a,b,label in [
            ('Sleep_Hours','Mood_Score','Sleep ↔ Mood'),
            ('Steps','Mood_Score','Steps ↔ Mood'),
            ('Study_Hours','Mood_Score','Study ↔ Mood'),
            ('Water_Intake_ml','Mood_Score','Hydration ↔ Mood')
        ]:
            if {a,b} <= set(df.columns):
                c = df[a].corr(df[b])
                if pd.notna(c):
                    insights.append(f"- {label} correlation: {c:.2f}")
        if {'Weekday','Mood_Score'} <= set(df.columns):
            wk = df.groupby('Weekday')['Mood_Score'].mean().sort_values(ascending=False)
            if len(wk) > 0:
                insights.append(f"- Highest average mood day: {wk.index[0]}")
        if insights:
            st.markdown("\n".join(insights))
        else:
            st.write("Not enough data for auto insights.")

        st.markdown("### Correlation Matrix")
        if st.button("Show Correlation Matrix"):
            num_cols = df.select_dtypes(include=['float64','int64']).columns
            if len(num_cols) > 1:
                corr = df[num_cols].corr().round(2)
                st.dataframe(corr, use_container_width=True)
            else:
                st.warning("Not enough numeric columns to generate correlation matrix.")

with tab4:
    st.subheader("Recommendations")
    if df is None:
        st.info("Upload CSV to get recommendations.")
    else:
        df = parse_features(df)
        st.markdown("### Dataset-level suggestions")
        for r in dataset_recommendations(df):
            st.write("- " + r)

    st.markdown("### Your personalized tips")
    c1,c2,c3 = st.columns(3)
    sleep_v = c1.slider("Sleep Hours", 0.0, 14.0, 6.0, 0.5)
    steps_v = c2.number_input("Steps", 0, 50000, 4500, 100)
    water_v = c3.number_input("Water Intake (ml)", 0, 10000, 1800, 50)
    study_v = c1.slider("Study Hours", 0.0, 16.0, 3.0, 0.5)
    wake_v = c2.slider("Wake-up Hour (0–23.99)", 0.0, 23.99, 9.0, 0.25)
    mood_v = c3.slider("Mood Score (1–10)", 1, 10, 6, 1)
    tips = recommend_for_values(sleep_v, steps_v, water_v, study_v, wake_v, mood_v)
    if tips:
        for t in tips:
            st.write("- " + t)
    else:
        st.write("Looks good — maintain consistency!")

    st.markdown("### ⬇ Export Report (.txt)")
    if df is not None and st.button("Download Analysis Report (.txt)"):
        buffer = io.StringIO()
        buffer.write("Daily Lifestyle & Wellness Report\n")
        buffer.write("---------------------------------\n\n")
        buffer.write("📊 Dataset Averages:\n")
        for col,label in [('Sleep_Hours','Sleep_Hours'),('Steps','Steps'),
                          ('Water_Intake_ml','Water (ml)'),('Study_Hours','Study_Hours'),
                          ('Mood_Score','Mood_Score')]:
            if col in df:
                buffer.write(f"- {label}: {df[col].mean():.2f}\n")
        buffer.write("\n🔎 Correlation (numeric columns):\n")
        num_cols = df.select_dtypes(include=['float64','int64']).columns
        if len(num_cols) > 1:
            corr = df[num_cols].corr().round(2)
            buffer.write(str(corr))
        else:
            buffer.write("Not enough numeric columns for correlation.\n")
        buffer.write("\n\n📝 Recommendations:\n")
        for r in dataset_recommendations(df):
            buffer.write(f"- {r}\n")
        st.download_button(label="⬇ Download TXT Report",
                           data=buffer.getvalue(),
                           file_name="wellness_report.txt",
                           mime="text/plain")

