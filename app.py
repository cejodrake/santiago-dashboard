import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Santiago · Behavior Dashboard",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;900&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
  }
  .main { background: #f0f4ff; }
  
  .hero-card {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecf8e 100%);
    border-radius: 20px;
    padding: 2rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
  }
  .hero-card h1 { font-size: 2.2rem; font-weight: 900; margin: 0; }
  .hero-card p  { font-size: 1rem; opacity: 0.9; margin: 0.3rem 0 0; }

  .metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 5px solid;
  }
  .metric-card.good   { border-color: #3ecf8e; }
  .metric-card.medium { border-color: #f6a623; }
  .metric-card.low    { border-color: #ff6b6b; }
  .metric-card h2 { font-family: 'Space Mono', monospace; font-size: 2rem; margin: 0; }
  .metric-card p  { color: #888; margin: 0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

  .section-title {
    font-size: 1.3rem; font-weight: 800; color: #2d3142;
    margin: 1.5rem 0 0.8rem;
    display: flex; align-items: center; gap: 0.5rem;
  }

  .insight-box {
    background: #fff8e7;
    border: 1.5px solid #f6a623;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }
  .insight-box.good {
    background: #edfff6; border-color: #3ecf8e;
  }
  .insight-box.alert {
    background: #fff0f0; border-color: #ff6b6b;
  }
  
  .pred-box {
    background: linear-gradient(135deg, #6c63ff22, #3ecf8e22);
    border: 2px solid #6c63ff55;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-box .big { font-size: 3rem; }
  .pred-box h3   { font-size: 1.5rem; font-weight: 900; margin: 0.3rem 0 0; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
SHEET_ID  = "1U76gI-6iKimWfkypUxtg4MtqvPr12lmJ77AiOjm19HI"
SHEET_GID = "1941134059"
CSV_URL   = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"

@st.cache_data(ttl=300)   # re-fetch every 5 min
def load_data():
    df = pd.read_csv(CSV_URL)
    df.columns = df.columns.str.strip()

    # Clean & type-cast
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')
    numeric_cols = ['follow_num','task_transition_num','safe_body_num','bonus_point','Total']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['behavior_level'] = df['behavior_level'].str.strip().str.lower()
    df['Activity']       = df['Activity'].str.strip()
    df = df.dropna(subset=['Total','behavior_level'])

    # Feature engineering
    df['day_of_week'] = df['Date'].dt.day_name()
    df['week_num']    = df['Date'].dt.isocalendar().week.astype(float)
    df['month']       = df['Date'].dt.month
    df['score_pct']   = (df['Total'] / 8 * 100).round(1)
    return df

df = load_data()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filtros")
    activities = ["Todas"] + sorted(df['Activity'].dropna().unique().tolist())
    sel_act = st.selectbox("Actividad", activities)
    
    levels = st.multiselect("Nivel de comportamiento",
                            ['good','medium','low'],
                            default=['good','medium','low'])
    
    date_range = st.date_input("Rango de fechas",
                               value=[df['Date'].min().date(), df['Date'].max().date()],
                               min_value=df['Date'].min().date(),
                               max_value=df['Date'].max().date())
    
    st.markdown("---")
    st.markdown("### 📊 Sobre este dashboard")
    st.markdown("Datos de comportamiento diario de **Santiago Calix** registrados por la escuela.")
    st.markdown("Se actualiza automáticamente desde Google Sheets.")

# ─── APPLY FILTERS ─────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_act != "Todas":
    filtered = filtered[filtered['Activity'] == sel_act]
filtered = filtered[filtered['behavior_level'].isin(levels)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['Date'].dt.date >= date_range[0]) &
        (filtered['Date'].dt.date <= date_range[1])
    ]

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1>🌟 Santiago · Behavior Dashboard</h1>
  <p>Seguimiento de comportamiento escolar · Datos en tiempo real desde Google Sheets</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ───────────────────────────────────────────────────────────────
total_rows = len(filtered)
avg_score  = filtered['Total'].mean()
good_pct   = (filtered['behavior_level'] == 'good').mean() * 100
best_act   = (filtered.groupby('Activity')['Total'].mean().idxmax()
              if not filtered.empty else "N/A")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card good">
      <p>Puntaje Promedio</p><h2>{avg_score:.1f}/8</h2></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card good">
      <p>Sesiones "Good"</p><h2>{good_pct:.0f}%</h2></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card medium">
      <p>Registros totales</p><h2>{total_rows}</h2></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card good">
      <p>Mejor actividad</p><h2 style="font-size:1.1rem;margin-top:.3rem">{best_act}</h2></div>""", unsafe_allow_html=True)

# ─── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Tendencias", "🏫 Por Actividad", "🤖 Modelo ML", "📋 Datos"])

# ═══════════════════════════════════════════════════════
# TAB 1 – TRENDS
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📈 Puntaje Total por Fecha</div>', unsafe_allow_html=True)

    daily = (filtered.groupby('Date')['Total']
             .mean().reset_index().sort_values('Date'))

    if not daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['Total'],
            mode='lines+markers',
            line=dict(color='#6c63ff', width=3),
            marker=dict(size=8, color='#6c63ff'),
            name='Puntaje diario'
        ))
        # Trend line
        if len(daily) > 2:
            x_num = np.arange(len(daily))
            lr = LinearRegression().fit(x_num.reshape(-1,1), daily['Total'])
            trend = lr.predict(x_num.reshape(-1,1))
            fig.add_trace(go.Scatter(
                x=daily['Date'], y=trend,
                mode='lines', line=dict(color='#3ecf8e', dash='dash', width=2),
                name='Tendencia'
            ))
        fig.update_layout(
            height=340, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation='h', y=1.1),
            yaxis=dict(range=[0,8.5], title='Puntaje (0-8)'),
            xaxis_title='Fecha'
        )
        st.plotly_chart(fig, use_container_width=True)

    # By day of week
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📅 Por Día de la Semana</div>', unsafe_allow_html=True)
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday']
        dow = (filtered.groupby('day_of_week')['Total'].mean()
               .reindex(dow_order).dropna().reset_index())
        dow.columns = ['Día','Promedio']
        fig2 = px.bar(dow, x='Día', y='Promedio',
                      color='Promedio', color_continuous_scale=['#ff6b6b','#f6a623','#3ecf8e'],
                      range_color=[0,8])
        fig2.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=5,r=5,t=5,b=5), showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🎯 Distribución de Niveles</div>', unsafe_allow_html=True)
        level_counts = filtered['behavior_level'].value_counts().reset_index()
        level_counts.columns = ['Nivel','Count']
        color_map = {'good':'#3ecf8e','medium':'#f6a623','low':'#ff6b6b'}
        fig3 = px.pie(level_counts, names='Nivel', values='Count',
                      color='Nivel', color_discrete_map=color_map,
                      hole=0.4)
        fig3.update_layout(height=280, paper_bgcolor='white',
                           margin=dict(l=5,r=5,t=5,b=5))
        st.plotly_chart(fig3, use_container_width=True)

    # Score breakdown
    st.markdown('<div class="section-title">🔍 Desglose de Puntajes por Categoría</div>', unsafe_allow_html=True)
    cats = ['follow_num','task_transition_num','safe_body_num','bonus_point']
    labels = ['Seguir Instrucciones','Transición de Tareas','Cuerpo Seguro','Bonus']
    avg_cats = [filtered[c].mean() for c in cats]
    fig4 = go.Figure(go.Bar(
        x=labels, y=avg_cats,
        marker_color=['#6c63ff','#3ecf8e','#f6a623','#ff6b6b'],
        text=[f"{v:.2f}" for v in avg_cats], textposition='outside'
    ))
    fig4.update_layout(height=280, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=5,r=5,t=5,b=5),
                       yaxis=dict(range=[0,2.5], title='Puntaje promedio (0-2)'))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 – BY ACTIVITY
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🏫 Puntaje Promedio por Actividad</div>', unsafe_allow_html=True)

    act_stats = (filtered.groupby('Activity')
                 .agg(Promedio=('Total','mean'),
                      Sesiones=('Total','count'),
                      Std=('Total','std'))
                 .reset_index().sort_values('Promedio', ascending=True))

    fig5 = go.Figure(go.Bar(
        y=act_stats['Activity'], x=act_stats['Promedio'],
        orientation='h',
        marker=dict(
            color=act_stats['Promedio'],
            colorscale=[[0,'#ff6b6b'],[0.5,'#f6a623'],[1,'#3ecf8e']],
            cmin=0, cmax=8
        ),
        text=[f"{v:.1f}" for v in act_stats['Promedio']],
        textposition='outside'
    ))
    fig5.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=10,r=40,t=10,b=10),
                       xaxis=dict(range=[0,9], title='Puntaje Promedio (0-8)'))
    st.plotly_chart(fig5, use_container_width=True)

    # Heatmap: activity vs day
    st.markdown('<div class="section-title">🗓️ Mapa de Calor: Actividad × Día</div>', unsafe_allow_html=True)
    heat_data = (filtered.groupby(['Activity','day_of_week'])['Total']
                 .mean().unstack(fill_value=np.nan))
    dow_cols = [d for d in ['Monday','Tuesday','Wednesday','Thursday','Friday'] if d in heat_data.columns]
    heat_data = heat_data[dow_cols]

    if not heat_data.empty:
        fig6 = px.imshow(heat_data,
                         color_continuous_scale=['#ff6b6b','#f6a623','#3ecf8e'],
                         zmin=0, zmax=8,
                         aspect='auto',
                         labels=dict(color='Puntaje'))
        fig6.update_layout(height=350, paper_bgcolor='white',
                           margin=dict(l=5,r=5,t=5,b=5))
        st.plotly_chart(fig6, use_container_width=True)

    # Table
    st.markdown('<div class="section-title">📊 Estadísticas por Actividad</div>', unsafe_allow_html=True)
    act_table = act_stats.copy()
    act_table['Promedio'] = act_table['Promedio'].round(2)
    act_table['Std']      = act_table['Std'].round(2)
    act_table = act_table.sort_values('Promedio', ascending=False)
    act_table.columns = ['Actividad','Prom. Puntaje','# Sesiones','Desv. Estándar']
    st.dataframe(act_table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# TAB 3 – ML MODEL
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🤖 Modelo de Predicción de Comportamiento</div>', unsafe_allow_html=True)

    # Prepare ML data
    ml_df = df.copy().dropna(subset=['Total','behavior_level','Activity'])
    le_act = LabelEncoder()
    le_day = LabelEncoder()
    ml_df['activity_enc'] = le_act.fit_transform(ml_df['Activity'])
    ml_df['day_enc']      = le_day.fit_transform(ml_df['day_of_week'].fillna('Unknown'))

    features = ['follow_num','task_transition_num','safe_body_num','bonus_point','activity_enc','day_enc']
    X = ml_df[features].fillna(0)
    y_reg = ml_df['Total']
    y_clf = ml_df['behavior_level']

    col_m1, col_m2 = st.columns(2)

    # ── Linear Regression ──
    with col_m1:
        st.markdown("#### 📉 Regresión Lineal (Puntaje Total)")
        if len(X) > 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
            lr = LinearRegression().fit(X_tr, y_tr)
            r2 = lr.score(X_te, y_te)
            y_pred = lr.predict(X_te)

            st.markdown(f"""<div class="pred-box">
              <div class="big">📈</div>
              <h3>R² = {r2:.3f}</h3>
              <p style="color:#555">El modelo explica el <b>{r2*100:.1f}%</b> de la variación<br>en el puntaje de Santiago</p>
            </div>""", unsafe_allow_html=True)

            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                x=y_te, y=y_pred, mode='markers',
                marker=dict(color='#6c63ff', opacity=0.7, size=8),
                name='Pred vs Real'
            ))
            fig_lr.add_trace(go.Scatter(
                x=[0,8], y=[0,8], mode='lines',
                line=dict(color='#3ecf8e', dash='dash'), name='Ideal'
            ))
            fig_lr.update_layout(
                height=280, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=5,r=5,t=20,b=5),
                xaxis_title='Real', yaxis_title='Predicho',
                title='Predicho vs Real'
            )
            st.plotly_chart(fig_lr, use_container_width=True)

            # Coefficients
            coef_df = pd.DataFrame({
                'Variable': ['Seguir Inst.','Transición','Cuerpo Seg.','Bonus','Actividad','Día Semana'],
                'Coeficiente': lr.coef_
            }).sort_values('Coeficiente', ascending=True)
            fig_coef = px.bar(coef_df, x='Coeficiente', y='Variable',
                              orientation='h',
                              color='Coeficiente',
                              color_continuous_scale=['#ff6b6b','white','#3ecf8e'],
                              color_continuous_midpoint=0)
            fig_coef.update_layout(height=220, plot_bgcolor='white', paper_bgcolor='white',
                                   margin=dict(l=5,r=5,t=20,b=5),
                                   coloraxis_showscale=False,
                                   title='Importancia de Variables')
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Se necesitan más datos para entrenar el modelo.")

    # ── Classification (Random Forest) ──
    with col_m2:
        st.markdown("#### 🌲 Clasificador (good / medium / low)")
        if len(X) > 10:
            le_lv = LabelEncoder()
            y_enc = le_lv.fit_transform(y_clf)
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr2, y_tr2)
            acc = accuracy_score(y_te2, rf.predict(X_te2))

            st.markdown(f"""<div class="pred-box">
              <div class="big">🌲</div>
              <h3>Exactitud = {acc*100:.1f}%</h3>
              <p style="color:#555">Random Forest predice el nivel<br>de comportamiento de Santiago</p>
            </div>""", unsafe_allow_html=True)

            # Feature importance
            fi = pd.DataFrame({
                'Variable': ['Seguir Inst.','Transición','Cuerpo Seg.','Bonus','Actividad','Día Semana'],
                'Importancia': rf.feature_importances_
            }).sort_values('Importancia')
            fig_fi = px.bar(fi, x='Importancia', y='Variable',
                            orientation='h',
                            color='Importancia',
                            color_continuous_scale=['#f0f0f0','#6c63ff'])
            fig_fi.update_layout(height=220, plot_bgcolor='white', paper_bgcolor='white',
                                 margin=dict(l=5,r=5,t=20,b=5),
                                 coloraxis_showscale=False,
                                 title='¿Qué variable importa más?')
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Predictor ──
    st.markdown("---")
    st.markdown('<div class="section-title">🔮 Predecir Comportamiento de Santiago</div>', unsafe_allow_html=True)
    st.markdown("Ingresa valores para una sesión y el modelo predecirá el resultado:")

    p1, p2, p3, p4 = st.columns(4)
    with p1: v_follow = st.slider("Seguir instrucciones", 0, 2, 1)
    with p2: v_trans  = st.slider("Transición de tareas", 0, 2, 1)
    with p3: v_safe   = st.slider("Cuerpo seguro", 0, 2, 1)
    with p4: v_bonus  = st.slider("Bonus", 0, 2, 1)

    p5, p6 = st.columns(2)
    with p5: v_act = st.selectbox("Actividad", sorted(df['Activity'].dropna().unique()))
    with p6: v_day = st.selectbox("Día de la semana", ['Monday','Tuesday','Wednesday','Thursday','Friday'])

    if st.button("🔮 Predecir", use_container_width=True):
        try:
            act_e = le_act.transform([v_act])[0]
            day_e = le_day.transform([v_day])[0]
        except:
            act_e, day_e = 0, 0

        X_pred = np.array([[v_follow, v_trans, v_safe, v_bonus, act_e, day_e]])
        pred_score = lr.predict(X_pred)[0]
        pred_class = le_lv.inverse_transform(rf.predict(X_pred))[0]
        proba      = rf.predict_proba(X_pred)[0]

        emoji = {'good':'🟢','medium':'🟡','low':'🔴'}.get(pred_class, '⚪')
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            st.markdown(f"""<div class="pred-box">
              <div class="big">{emoji}</div>
              <h3>{pred_class.upper()}</h3>
              <p>Nivel predicho</p></div>""", unsafe_allow_html=True)
        with c_b:
            st.markdown(f"""<div class="pred-box">
              <div class="big">📊</div>
              <h3>{max(0,min(8,pred_score)):.1f} / 8</h3>
              <p>Puntaje estimado</p></div>""", unsafe_allow_html=True)
        with c_c:
            classes = le_lv.classes_
            prob_df = pd.DataFrame({'Nivel': classes, 'Probabilidad': proba})
            fig_p = px.bar(prob_df, x='Nivel', y='Probabilidad',
                           color='Nivel',
                           color_discrete_map={'good':'#3ecf8e','medium':'#f6a623','low':'#ff6b6b'})
            fig_p.update_layout(height=180, plot_bgcolor='white', paper_bgcolor='white',
                                margin=dict(l=5,r=5,t=5,b=5), showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4 – RAW DATA
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📋 Datos Completos</div>', unsafe_allow_html=True)
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
    st.download_button("⬇️ Descargar CSV", csv, "santiago_behavior.csv", "text/csv")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>Dashboard de seguimiento para Santiago Calix · "
    "Datos actualizados desde Google Sheets cada 5 minutos</p>",
    unsafe_allow_html=True
)
