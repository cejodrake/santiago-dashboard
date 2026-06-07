import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Santiago · Behavior Dashboard",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "en": {
        "lang_label":        "🌐 Language",
        "filters":           "🔍 Filters",
        "activity":          "Activity",
        "all":               "All",
        "behavior_level":    "Behavior Level",
        "date_range":        "Date Range",
        "about":             "### 📊 About this Dashboard",
        "about_text":        "Daily behavior data for **Santiago Calix** recorded by his school. Auto-updates from Google Sheets.",
        "hero_title":        "🌟 Santiago · Behavior Dashboard",
        "hero_sub":          "Daily school behavior tracking · Live data from Google Sheets",
        "avg_score":         "Avg Score",
        "good_sessions":     "\"Good\" Sessions",
        "low_sessions":      "\"Low\" Sessions",
        "best_activity":     "Best Activity",
        "tab_trends":        "📈 Trends",
        "tab_activity":      "🏫 By Activity",
        "tab_ml":            "🤖 ML Model",
        "tab_data":          "📋 Data",
        # Tab 1
        "score_by_date":     "📈 Total Score Over Time",
        "daily_score":       "Daily Score",
        "trend":             "Trend",
        "score_08":          "Score (0-8)",
        "date":              "Date",
        "by_weekday":        "📅 By Day of Week",
        "level_dist":        "🎯 Level Distribution",
        "score_breakdown":   "🔍 Score Breakdown by Category",
        "avg_score_02":      "Average Score (0-2)",
        "follow":            "Follow Instructions",
        "transition":        "Task Transition",
        "safe_body":         "Safe Body",
        "bonus":             "Bonus Points",
        # Tab 2
        "low_by_act":        "🔴 % LOW Behavior by Activity",
        "pct_low_sessions":  "% LOW Sessions",
        "full_dist":         "📊 Full Distribution by Activity",
        "avg_by_act":        "⭐ Average Score by Activity",
        "avg_score_label":   "Average Score (0-8)",
        "summary_table":     "📋 Summary Table",
        "activity_col":      "Activity",
        "avg_score_col":     "Avg Score",
        "sessions_col":      "# Sessions",
        "std_col":           "Std Dev",
        # Tab 3
        "ml_title":          "🤖 Behavior Prediction Model",
        "linear_reg":        "📉 Linear Regression (Total Score)",
        "model_explains":    "The model explains",
        "of_variation":      "of the variation in Santiago's score",
        "predicted_vs_real": "Predicted vs Actual",
        "actual":            "Actual",
        "predicted":         "Predicted",
        "ideal":             "Ideal",
        "variable_weight":   "Variable Weight",
        "need_more_data":    "More data needed to train the model.",
        "rf_title":          "🌲 Random Forest Classifier (good / medium / low)",
        "rf_accuracy":       "Accuracy",
        "rf_subtitle":       "Random Forest predicts Santiago's behavior level",
        "feature_importance":"What matters most?",
        "predictor_title":   "🔮 Predict Santiago's Behavior",
        "predictor_sub":     "Enter values for a session and the model will predict the outcome:",
        "follow_slider":     "Follow Instructions",
        "trans_slider":      "Task Transition",
        "safe_slider":       "Safe Body",
        "bonus_slider":      "Bonus",
        "activity_sel":      "Activity",
        "day_sel":           "Day of Week",
        "predict_btn":       "🔮 Predict",
        "predicted_level":   "Predicted Level",
        "estimated_score":   "Estimated Score",
        # Tab 4
        "full_data":         "📋 Full Dataset",
        "download_csv":      "⬇️ Download CSV",
        "footer":            "Behavior tracking dashboard for Santiago Calix · Auto-updates from Google Sheets every 5 minutes",
        "var_follow":        "Follow Inst.",
        "var_trans":         "Transition",
        "var_safe":          "Safe Body",
        "var_bonus":         "Bonus",
        "var_activity":      "Activity",
        "var_day":           "Day of Week",
    },
    "es": {
        "lang_label":        "🌐 Idioma",
        "filters":           "🔍 Filtros",
        "activity":          "Actividad",
        "all":               "Todas",
        "behavior_level":    "Nivel de comportamiento",
        "date_range":        "Rango de fechas",
        "about":             "### 📊 Sobre este dashboard",
        "about_text":        "Datos de comportamiento diario de **Santiago Calix** registrados por la escuela. Se actualiza automáticamente desde Google Sheets.",
        "hero_title":        "🌟 Santiago · Dashboard de Comportamiento",
        "hero_sub":          "Seguimiento de comportamiento escolar · Datos en tiempo real desde Google Sheets",
        "avg_score":         "Puntaje Promedio",
        "good_sessions":     "Sesiones \"Good\"",
        "low_sessions":      "Sesiones \"Low\"",
        "best_activity":     "Mejor Actividad",
        "tab_trends":        "📈 Tendencias",
        "tab_activity":      "🏫 Por Actividad",
        "tab_ml":            "🤖 Modelo ML",
        "tab_data":          "📋 Datos",
        # Tab 1
        "score_by_date":     "📈 Puntaje Total por Fecha",
        "daily_score":       "Puntaje diario",
        "trend":             "Tendencia",
        "score_08":          "Puntaje (0-8)",
        "date":              "Fecha",
        "by_weekday":        "📅 Por Día de la Semana",
        "level_dist":        "🎯 Distribución de Niveles",
        "score_breakdown":   "🔍 Desglose por Categoría",
        "avg_score_02":      "Puntaje promedio (0-2)",
        "follow":            "Seguir Instrucciones",
        "transition":        "Transición de Tareas",
        "safe_body":         "Cuerpo Seguro",
        "bonus":             "Puntos Bonus",
        # Tab 2
        "low_by_act":        "🔴 % de Comportamiento LOW por Actividad",
        "pct_low_sessions":  "% Sesiones LOW",
        "full_dist":         "📊 Distribución Completa por Actividad",
        "avg_by_act":        "⭐ Puntaje Promedio por Actividad",
        "avg_score_label":   "Puntaje Promedio (0-8)",
        "summary_table":     "📋 Tabla Resumen",
        "activity_col":      "Actividad",
        "avg_score_col":     "Prom. Puntaje",
        "sessions_col":      "# Sesiones",
        "std_col":           "Desv. Estándar",
        # Tab 3
        "ml_title":          "🤖 Modelo de Predicción de Comportamiento",
        "linear_reg":        "📉 Regresión Lineal (Puntaje Total)",
        "model_explains":    "El modelo explica el",
        "of_variation":      "de la variación en el puntaje de Santiago",
        "predicted_vs_real": "Predicho vs Real",
        "actual":            "Real",
        "predicted":         "Predicho",
        "ideal":             "Ideal",
        "variable_weight":   "Peso de cada variable",
        "need_more_data":    "Se necesitan más datos para entrenar el modelo.",
        "rf_title":          "🌲 Clasificador Random Forest (good / medium / low)",
        "rf_accuracy":       "Exactitud",
        "rf_subtitle":       "Random Forest predice el nivel de comportamiento de Santiago",
        "feature_importance":"¿Qué variable importa más?",
        "predictor_title":   "🔮 Predecir Comportamiento de Santiago",
        "predictor_sub":     "Ingresa valores para una sesión y el modelo predecirá el resultado:",
        "follow_slider":     "Seguir instrucciones",
        "trans_slider":      "Transición de tareas",
        "safe_slider":       "Cuerpo seguro",
        "bonus_slider":      "Bonus",
        "activity_sel":      "Actividad",
        "day_sel":           "Día de la semana",
        "predict_btn":       "🔮 Predecir",
        "predicted_level":   "Nivel predicho",
        "estimated_score":   "Puntaje estimado",
        # Tab 4
        "full_data":         "📋 Datos Completos",
        "download_csv":      "⬇️ Descargar CSV",
        "footer":            "Dashboard de seguimiento para Santiago Calix · Datos actualizados desde Google Sheets cada 5 minutos",
        "var_follow":        "Seguir Inst.",
        "var_trans":         "Transición",
        "var_safe":          "Cuerpo Seg.",
        "var_bonus":         "Bonus",
        "var_activity":      "Actividad",
        "var_day":           "Día Semana",
    }
}

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;900&family=Space+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
  .main { background: #f0f4ff; }
  .hero-card {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecf8e 100%);
    border-radius: 20px; padding: 2rem; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(108,99,255,0.25);
  }
  .hero-card h1 { font-size: 2.2rem; font-weight: 900; margin: 0; }
  .hero-card p  { font-size: 1rem; opacity: 0.9; margin: 0.3rem 0 0; }
  .metric-card {
    background: white; border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 6px solid;
  }
  .metric-card.good   { border-color: #1db87a; }
  .metric-card.medium { border-color: #e69500; }
  .metric-card.low    { border-color: #e63946; }
  .metric-card.info   { border-color: #6c63ff; }
  .metric-card h2 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; margin: 0.2rem 0 0; color: #1a1a2e;
  }
  .metric-card h2.good-num   { color: #0e7a50; }
  .metric-card h2.medium-num { color: #a86800; }
  .metric-card h2.low-num    { color: #b5202b; }
  .metric-card h2.info-num   { color: #4a42cc; }
  .metric-card p { color: #666; margin: 0; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
  .section-title {
    font-size: 1.3rem; font-weight: 800; color: #2d3142;
    margin: 1.5rem 0 0.8rem; display: flex; align-items: center; gap: 0.5rem;
  }
  .pred-box {
    background: linear-gradient(135deg,#6c63ff18,#3ecf8e18);
    border: 2px solid #6c63ff44; border-radius: 16px;
    padding: 1.5rem; text-align: center;
  }
  .pred-box .big { font-size: 3rem; }
  .pred-box h3   { font-size: 1.5rem; font-weight: 900; color: #1a1a2e; margin: 0.3rem 0 0; }
  /* language toggle button styling */
  div[data-testid="stRadio"] > label { font-weight: 700; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
SHEET_ID  = "1U76gI-6iKimWfkypUxtg4MtqvPr12lmJ77AiOjm19HI"
SHEET_GID = "1941134059"
CSV_URL   = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(CSV_URL)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')
    for c in ['follow_num','task_transition_num','safe_body_num','bonus_point','Total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['behavior_level'] = df['behavior_level'].str.strip().str.lower()
    df['Activity']       = df['Activity'].str.strip()
    df = df.dropna(subset=['Total','behavior_level'])
    df['day_of_week'] = df['Date'].dt.day_name()
    df['week_num']    = df['Date'].dt.isocalendar().week.astype(float)
    df['month']       = df['Date'].dt.month
    df['score_pct']   = (df['Total'] / 8 * 100).round(1)
    return df

df = load_data()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language selector — default English
    lang = st.radio("🌐 Language / Idioma", ["English", "Español"],
                    index=0, horizontal=True)
    L = T["en"] if lang == "English" else T["es"]

    st.markdown("---")
    st.markdown(f"### {L['filters']}")
    activities = [L["all"]] + sorted(df['Activity'].dropna().unique().tolist())
    sel_act    = st.selectbox(L["activity"], activities)
    levels     = st.multiselect(L["behavior_level"], ['good','medium','low'],
                                default=['good','medium','low'])
    date_range = st.date_input(L["date_range"],
                               value=[df['Date'].min().date(), df['Date'].max().date()],
                               min_value=df['Date'].min().date(),
                               max_value=df['Date'].max().date())
    st.markdown("---")
    st.markdown(L["about"])
    st.markdown(L["about_text"])

# ─── FILTERS ───────────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_act != L["all"]:
    filtered = filtered[filtered['Activity'] == sel_act]
filtered = filtered[filtered['behavior_level'].isin(levels)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['Date'].dt.date >= date_range[0]) &
        (filtered['Date'].dt.date <= date_range[1])
    ]

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-card">
  <h1>{L['hero_title']}</h1>
  <p>{L['hero_sub']}</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ─────────────────────────────────────────────────────────────────
avg_score = filtered['Total'].mean() if not filtered.empty else 0
good_pct  = (filtered['behavior_level'] == 'good').mean() * 100 if not filtered.empty else 0
low_pct   = (filtered['behavior_level'] == 'low').mean()  * 100 if not filtered.empty else 0
best_act  = (filtered.groupby('Activity')['Total'].mean().idxmax()
             if not filtered.empty else "N/A")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card good">
      <p>{L['avg_score']}</p>
      <h2 class="good-num">{avg_score:.1f}<span style="font-size:1rem;color:#888"> /8</span></h2>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card good">
      <p>{L['good_sessions']}</p>
      <h2 class="good-num">{good_pct:.0f}<span style="font-size:1rem;color:#888">%</span></h2>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card low">
      <p>{L['low_sessions']}</p>
      <h2 class="low-num">{low_pct:.0f}<span style="font-size:1rem;color:#888">%</span></h2>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card info">
      <p>{L['best_activity']}</p>
      <h2 class="info-num" style="font-size:1rem;padding-top:.4rem">{best_act}</h2>
    </div>""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([L["tab_trends"], L["tab_activity"], L["tab_ml"], L["tab_data"]])

# ════════════════════════════════════════
# TAB 1 – TRENDS
# ════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-title">{L["score_by_date"]}</div>', unsafe_allow_html=True)

    daily = filtered.groupby('Date')['Total'].mean().reset_index().sort_values('Date')
    if not daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['Total'], mode='lines+markers',
            line=dict(color='#6c63ff', width=3), marker=dict(size=8, color='#6c63ff'),
            name=L["daily_score"]
        ))
        if len(daily) > 2:
            x_num = np.arange(len(daily))
            lr_t  = LinearRegression().fit(x_num.reshape(-1,1), daily['Total'])
            trend = lr_t.predict(x_num.reshape(-1,1))
            fig.add_trace(go.Scatter(
                x=daily['Date'], y=trend, mode='lines',
                line=dict(color='#3ecf8e', dash='dash', width=2), name=L["trend"]
            ))
        fig.update_layout(height=340, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=10,r=10,t=10,b=10),
                          legend=dict(orientation='h', y=1.1),
                          yaxis=dict(range=[0,8.5], title=L["score_08"]),
                          xaxis_title=L["date"])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="section-title">{L["by_weekday"]}</div>', unsafe_allow_html=True)
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday']
        dow = (filtered.groupby('day_of_week')['Total'].mean()
               .reindex(dow_order).dropna().reset_index())
        dow.columns = ['Day', 'Avg']
        fig2 = px.bar(dow, x='Day', y='Avg',
                      color='Avg',
                      color_continuous_scale=[[0,'#e63946'],[0.5,'#e69500'],[1,'#1db87a']],
                      range_color=[0,8], text='Avg')
        fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside',
                           textfont=dict(color='#1a1a2e', size=13, family='Space Mono'))
        fig2.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=5,r=5,t=5,b=5),
                           showlegend=False, coloraxis_showscale=False,
                           yaxis=dict(range=[0,9]))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown(f'<div class="section-title">{L["level_dist"]}</div>', unsafe_allow_html=True)
        level_counts = filtered['behavior_level'].value_counts().reset_index()
        level_counts.columns = ['Level','Count']
        color_map = {'good':'#1db87a','medium':'#e69500','low':'#e63946'}
        fig3 = px.pie(level_counts, names='Level', values='Count',
                      color='Level', color_discrete_map=color_map, hole=0.45)
        fig3.update_traces(textfont=dict(color='white', size=14, family='Nunito'))
        fig3.update_layout(height=300, paper_bgcolor='white', margin=dict(l=5,r=5,t=5,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f'<div class="section-title">{L["score_breakdown"]}</div>', unsafe_allow_html=True)
    cats      = ['follow_num','task_transition_num','safe_body_num','bonus_point']
    cat_names = [L["follow"], L["transition"], L["safe_body"], L["bonus"]]
    avg_cats  = [filtered[c].mean() for c in cats]
    fig4 = go.Figure(go.Bar(
        x=cat_names, y=avg_cats,
        marker_color=['#6c63ff','#1db87a','#e69500','#e63946'],
        text=[f"{v:.2f}" for v in avg_cats], textposition='outside',
        textfont=dict(color='#1a1a2e', size=13, family='Space Mono')
    ))
    fig4.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=5,r=5,t=5,b=5),
                       yaxis=dict(range=[0,2.6], title=L["avg_score_02"]))
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════
# TAB 2 – BY ACTIVITY
# ════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{L["low_by_act"]}</div>', unsafe_allow_html=True)

    if not filtered.empty and 'low' in filtered['behavior_level'].values:
        activity_pct = pd.crosstab(
            filtered['Activity'], filtered['behavior_level'], normalize='index') * 100
        for col in ['good','medium','low']:
            if col not in activity_pct.columns:
                activity_pct[col] = 0
        activity_pct = activity_pct.sort_values(by='low', ascending=True)

        fig_low = go.Figure(go.Bar(
            y=activity_pct.index, x=activity_pct['low'], orientation='h',
            marker=dict(color=activity_pct['low'],
                        colorscale=[[0,'#1db87a'],[0.5,'#e69500'],[1,'#e63946']],
                        cmin=0, cmax=100),
            text=[f"{v:.1f}%" for v in activity_pct['low']],
            textposition='outside',
            textfont=dict(color='#1a1a2e', size=13, family='Space Mono')
        ))
        fig_low.update_layout(
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10,r=60,t=10,b=10),
            xaxis=dict(range=[0,110], title=L["pct_low_sessions"], ticksuffix='%')
        )
        st.plotly_chart(fig_low, use_container_width=True)

        st.markdown(f'<div class="section-title">{L["full_dist"]}</div>', unsafe_allow_html=True)
        act_sorted = activity_pct.sort_values('good', ascending=True)
        fig_stack  = go.Figure()
        for level, color in [('low','#e63946'),('medium','#e69500'),('good','#1db87a')]:
            if level in act_sorted.columns:
                fig_stack.add_trace(go.Bar(
                    y=act_sorted.index, x=act_sorted[level],
                    name=level.capitalize(), orientation='h',
                    marker_color=color,
                    text=[f"{v:.0f}%" for v in act_sorted[level]],
                    textposition='inside', textfont=dict(color='white', size=11)
                ))
        fig_stack.update_layout(
            barmode='stack', height=420, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(title='%', ticksuffix='%'),
            legend=dict(orientation='h', y=1.05)
        )
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Not enough data for this view with the current filters."
                if lang == "English" else
                "No hay suficientes datos para esta vista con los filtros actuales.")

    st.markdown(f'<div class="section-title">{L["avg_by_act"]}</div>', unsafe_allow_html=True)
    act_stats = (filtered.groupby('Activity')
                 .agg(Avg=('Total','mean'), Sessions=('Total','count'), Std=('Total','std'))
                 .reset_index().sort_values('Avg', ascending=True))

    fig5 = go.Figure(go.Bar(
        y=act_stats['Activity'], x=act_stats['Avg'], orientation='h',
        marker=dict(color=act_stats['Avg'],
                    colorscale=[[0,'#e63946'],[0.5,'#e69500'],[1,'#1db87a']],
                    cmin=0, cmax=8),
        text=[f"{v:.1f}" for v in act_stats['Avg']],
        textposition='outside',
        textfont=dict(color='#1a1a2e', size=13, family='Space Mono')
    ))
    fig5.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=10,r=40,t=10,b=10),
                       xaxis=dict(range=[0,9], title=L["avg_score_label"]),
                       coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f'<div class="section-title">{L["summary_table"]}</div>', unsafe_allow_html=True)
    tbl = act_stats.copy()
    tbl['Avg'] = tbl['Avg'].round(2)
    tbl['Std'] = tbl['Std'].round(2)
    tbl = tbl.sort_values('Avg', ascending=False)
    tbl.columns = [L["activity_col"], L["avg_score_col"], L["sessions_col"], L["std_col"]]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 3 – ML MODEL
# ════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">{L["ml_title"]}</div>', unsafe_allow_html=True)

    ml_df = df.copy().dropna(subset=['Total','behavior_level','Activity'])
    le_act = LabelEncoder()
    le_day = LabelEncoder()
    ml_df['activity_enc'] = le_act.fit_transform(ml_df['Activity'])
    ml_df['day_enc']      = le_day.fit_transform(ml_df['day_of_week'].fillna('Unknown'))

    features = ['follow_num','task_transition_num','safe_body_num','bonus_point','activity_enc','day_enc']
    X     = ml_df[features].fillna(0)
    y_reg = ml_df['Total']
    y_clf = ml_df['behavior_level']

    var_labels = [L["var_follow"], L["var_trans"], L["var_safe"],
                  L["var_bonus"], L["var_activity"], L["var_day"]]

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(f"#### {L['linear_reg']}")
        if len(X) > 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
            lr = LinearRegression().fit(X_tr, y_tr)
            r2 = lr.score(X_te, y_te)
            y_pred_lr = lr.predict(X_te)

            st.markdown(f"""<div class="pred-box">
              <div class="big">📈</div>
              <h3>R² = {r2:.3f}</h3>
              <p style="color:#555">{L['model_explains']} <b>{r2*100:.1f}%</b> {L['of_variation']}</p>
            </div>""", unsafe_allow_html=True)

            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(x=y_te, y=y_pred_lr, mode='markers',
                                        marker=dict(color='#6c63ff', opacity=0.7, size=8),
                                        name=f"{L['predicted']} vs {L['actual']}"))
            fig_lr.add_trace(go.Scatter(x=[0,8], y=[0,8], mode='lines',
                                        line=dict(color='#1db87a', dash='dash'),
                                        name=L["ideal"]))
            fig_lr.update_layout(height=260, plot_bgcolor='white', paper_bgcolor='white',
                                 margin=dict(l=5,r=5,t=20,b=5),
                                 xaxis_title=L["actual"], yaxis_title=L["predicted"],
                                 title=L["predicted_vs_real"])
            st.plotly_chart(fig_lr, use_container_width=True)

            coef_df = pd.DataFrame({'Variable': var_labels, 'Coef': lr.coef_}).sort_values('Coef')
            fig_coef = px.bar(coef_df, x='Coef', y='Variable', orientation='h',
                              color='Coef',
                              color_continuous_scale=['#e63946','#f5f5f5','#1db87a'],
                              color_continuous_midpoint=0)
            fig_coef.update_traces(texttemplate='%{x:.2f}', textposition='outside',
                                   textfont=dict(color='#1a1a2e', size=11))
            fig_coef.update_layout(height=220, plot_bgcolor='white', paper_bgcolor='white',
                                   margin=dict(l=5,r=5,t=20,b=5),
                                   coloraxis_showscale=False, title=L["variable_weight"])
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info(L["need_more_data"])

    with col_m2:
        st.markdown(f"#### {L['rf_title']}")
        if len(X) > 10:
            le_lv = LabelEncoder()
            y_enc = le_lv.fit_transform(y_clf)
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr2, y_tr2)
            acc = accuracy_score(y_te2, rf.predict(X_te2))

            st.markdown(f"""<div class="pred-box">
              <div class="big">🌲</div>
              <h3>{L['rf_accuracy']} = {acc*100:.1f}%</h3>
              <p style="color:#555">{L['rf_subtitle']}</p>
            </div>""", unsafe_allow_html=True)

            fi = pd.DataFrame({'Variable': var_labels, 'Importance': rf.feature_importances_}).sort_values('Importance')
            fig_fi = px.bar(fi, x='Importance', y='Variable', orientation='h',
                            color='Importance',
                            color_continuous_scale=['#e8e8ff','#6c63ff'])
            fig_fi.update_traces(texttemplate='%{x:.3f}', textposition='outside',
                                 textfont=dict(color='#1a1a2e', size=11))
            fig_fi.update_layout(height=220, plot_bgcolor='white', paper_bgcolor='white',
                                 margin=dict(l=5,r=5,t=20,b=5),
                                 coloraxis_showscale=False, title=L["feature_importance"])
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Interactive Predictor ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-title">{L["predictor_title"]}</div>', unsafe_allow_html=True)
    st.markdown(L["predictor_sub"])

    if len(X) > 10:
        p1, p2, p3, p4 = st.columns(4)
        with p1: v_follow = st.slider(L["follow_slider"], 0, 2, 1)
        with p2: v_trans  = st.slider(L["trans_slider"],  0, 2, 1)
        with p3: v_safe   = st.slider(L["safe_slider"],   0, 2, 1)
        with p4: v_bonus  = st.slider(L["bonus_slider"],  0, 2, 1)

        p5, p6 = st.columns(2)
        with p5: v_act = st.selectbox(L["activity_sel"], sorted(df['Activity'].dropna().unique()))
        with p6: v_day = st.selectbox(L["day_sel"],
                                      ['Monday','Tuesday','Wednesday','Thursday','Friday'])

        if st.button(L["predict_btn"], use_container_width=True):
            try:
                act_e = le_act.transform([v_act])[0]
                day_e = le_day.transform([v_day])[0]
            except Exception:
                act_e, day_e = 0, 0

            X_pred     = np.array([[v_follow, v_trans, v_safe, v_bonus, act_e, day_e]])
            pred_score = lr.predict(X_pred)[0]
            pred_class = le_lv.inverse_transform(rf.predict(X_pred))[0]
            proba      = rf.predict_proba(X_pred)[0]

            emoji = {'good':'🟢','medium':'🟡','low':'🔴'}.get(pred_class,'⚪')
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.markdown(f"""<div class="pred-box">
                  <div class="big">{emoji}</div>
                  <h3>{pred_class.upper()}</h3>
                  <p>{L['predicted_level']}</p></div>""", unsafe_allow_html=True)
            with c_b:
                st.markdown(f"""<div class="pred-box">
                  <div class="big">📊</div>
                  <h3>{max(0,min(8,pred_score)):.1f} / 8</h3>
                  <p>{L['estimated_score']}</p></div>""", unsafe_allow_html=True)
            with c_c:
                classes = le_lv.classes_
                prob_df = pd.DataFrame({'Level': classes, 'Probability': proba})
                fig_p   = px.bar(prob_df, x='Level', y='Probability',
                                 color='Level',
                                 color_discrete_map={'good':'#1db87a','medium':'#e69500','low':'#e63946'})
                fig_p.update_traces(texttemplate='%{y:.0%}', textposition='outside',
                                    textfont=dict(color='#1a1a2e', size=12))
                fig_p.update_layout(height=200, plot_bgcolor='white', paper_bgcolor='white',
                                    margin=dict(l=5,r=5,t=5,b=5),
                                    showlegend=False, yaxis=dict(range=[0,1.15]))
                st.plotly_chart(fig_p, use_container_width=True)

# ════════════════════════════════════════
# TAB 4 – RAW DATA
# ════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-title">{L["full_data"]}</div>', unsafe_allow_html=True)
    show_cols = ['Date','Activity','Time','follow_num','task_transition_num',
                 'safe_body_num','bonus_point','Total','behavior_level','Notes']
    show_cols = [c for c in show_cols if c in filtered.columns]

    def color_level(val):
        colors = {'good':'#edfff6','medium':'#fff8e7','low':'#fff0f0'}
        return f"background-color: {colors.get(val,'white')}"

    styled = (filtered[show_cols]
              .sort_values('Date', ascending=False)
              .style.map(color_level, subset=['behavior_level']))
    st.dataframe(styled, use_container_width=True, height=500)

    csv = filtered[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(L["download_csv"], csv, "santiago_behavior.csv", "text/csv")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:#aaa;font-size:0.8rem;'>{L['footer']}</p>",
    unsafe_allow_html=True
)
