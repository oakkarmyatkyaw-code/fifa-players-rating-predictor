import streamlit as st
import pandas as pd
import numpy as np
import pickle

import streamlit as st

st.markdown("""
    <style>
    .stApp {
        background-image: url("app/static/Background.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)
# PAGE CONFIGURATION & HEADER
st.set_page_config(page_title="FIFA Player Rating Predictor", layout="wide")

st.title("⚽ FIFA Player Overall Rating Prediction ")
st.write("Enter player attributes below. Remaining attributes will be auto-filled with dataset averages.")


with st.sidebar:
    st.image('parami.jpg')
    st.text("Student Name: Oak Kar Myat Kyaw")
    st.text("Student ID: PIUS20230036")
    st.text("Class of 2027")

# STEP 1: LOAD PICKLE MODEL & RAW DATASET FOR DYNAMIC BOUNDARIES
@st.cache_resource
def load_fifa_model():
    """Reads binary pickled pipeline model from disk."""
    with open('Player_rating Predictor.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_fifa_model()
except FileNotFoundError:
    st.error("Model file 'fifa_rating_model.pkl' not found! Please ensure model pickle file exists.")

df = pd.read_csv("fifa_players.csv")
df.columns = df.columns.str.strip()
missing_values = df.isnull().sum()
mv_columns = missing_values[missing_values > 0].index.tolist()
mv_columns = [col for col in mv_columns if col != 'value_euro']
df = df.drop(columns=mv_columns)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('overall_rating').tolist()

# STEP 2: FLEXIBLE 2-COLUMN LAYOUT SETUP (14 INTERACTIVE FEATURES)
col1, col2 = st.columns(2)

def get_col_stats(col_name):
    return (
        float(df[col_name].min()), 
        float(df[col_name].max()), 
        float(df[col_name].mean())
    )

# COLUMN 1: FEATURES 1 TO 7
with col1:
    st.subheader("🏃 General & Passing Attributes")
    
    p_min, p_max, p_avg = get_col_stats('potential')
    var1_potential = st.slider("Potential", min_value=p_min, max_value=p_max, value=p_avg, step=1.0)
    
    v_min, v_max, v_avg = get_col_stats('value_euro')
    var2_value_euro = st.number_input("Value (€ Euro)", min_value=v_min, max_value=v_max, value=v_avg, step=100000.0)
    
    sm_min, sm_max, sm_avg = get_col_stats('skill_moves(1-5)')
    var3_skill_moves = st.selectbox("Skill Moves (1-5)", options=sorted(df['skill_moves(1-5)'].unique().tolist()))
    
    c_min, c_max, c_avg = get_col_stats('crossing')
    var4_crossing = st.slider("Crossing", min_value=c_min, max_value=c_max, value=c_avg, step=1.0)
    
    sp_min, sp_max, sp_avg = get_col_stats('short_passing')
    var5_short_passing = st.slider("Short Passing", min_value=sp_min, max_value=sp_max, value=sp_avg, step=1.0)
    
    d_min, d_max, d_avg = get_col_stats('dribbling')
    var6_dribbling = st.slider("Dribbling", min_value=d_min, max_value=d_max, value=d_avg, step=1.0)
    
    crv_min, crv_max, crv_avg = get_col_stats('curve')
    var7_curve = st.slider("Curve", min_value=crv_min, max_value=crv_max, value=crv_avg, step=1.0)

# COLUMN 2: FEATURES 8 TO 14
with col2:
    st.subheader("🎯 Shooting & Control Attributes")
    
    lp_min, lp_max, lp_avg = get_col_stats('long_passing')
    var8_long_passing = st.slider("Long Passing", min_value=lp_min, max_value=lp_max, value=lp_avg, step=1.0)
    
    bc_min, bc_max, bc_avg = get_col_stats('ball_control')
    var9_ball_control = st.slider("Ball Control", min_value=bc_min, max_value=bc_avg, value=bc_avg, step=1.0)
    
    r_min, r_max, r_avg = get_col_stats('reactions')
    var10_reactions = st.slider("Reactions", min_value=r_min, max_value=r_max, value=r_avg, step=1.0)
    
    shp_min, shp_max, shp_avg = get_col_stats('shot_power')
    var11_shot_power = st.slider("Shot Power", min_value=shp_min, max_value=shp_max, value=shp_avg, step=1.0)
    
    ls_min, ls_max, ls_avg = get_col_stats('long_shots')
    var12_long_shots = st.slider("Long Shots", min_value=ls_min, max_value=ls_max, value=ls_avg, step=1.0)
    
    vis_min, vis_max, vis_avg = get_col_stats('vision')
    var13_vision = st.slider("Vision", min_value=vis_min, max_value=vis_max, value=vis_avg, step=1.0)
    
    cmp_min, cmp_max, cmp_avg = get_col_stats('composure')
    var14_composure = st.slider("Composure", min_value=cmp_min, max_value=cmp_max, value=cmp_avg, step=1.0)

# STEP 3: PREDICTION EXECUTION (AUTO-FILL ALL NUMERICAL COLUMNS)
st.markdown("---")
if st.button("🚀 Calculate Estimated Overall Rating", use_container_width=True):
    
    interactive_inputs = {
        'potential': var1_potential,
        'value_euro': var2_value_euro,
        'skill_moves(1-5)': var3_skill_moves,
        'crossing': var4_crossing,
        'short_passing': var5_short_passing,
        'dribbling': var6_dribbling,
        'curve': var7_curve,
        'long_passing': var8_long_passing,
        'ball_control': var9_ball_control,
        'reactions': var10_reactions,
        'shot_power': var11_shot_power,
        'long_shots': var12_long_shots,
        'vision': var13_vision,
        'composure': var14_composure
    }
    

    full_input_dict = {}
    for col in num_cols:
        if col in interactive_inputs:
            full_input_dict[col] = interactive_inputs[col]
        else:
            full_input_dict[col] = float(df[col].mean())
            
    
    user_input_df = pd.DataFrame([full_input_dict])[num_cols]
    
    try:
        predicted_rating = model.predict(user_input_df)[0]
        st.balloons()
        st.success(f"### ⚽ Estimated FIFA Overall Rating: **{predicted_rating:.2f}** / 100")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

