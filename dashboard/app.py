"""
AI-Based Drug Stock & Supply Chain Optimization System
Interactive Streamlit Dashboard
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

# Database & backend services fallback
from backend.database import SessionLocal, init_db
from backend import crud, schemas, models
from backend.services.forecasting_service import generate_drug_forecast
from backend.services.optimization_service import optimize_reorder_quantities

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PharmaAI - Drug Stock & Supply Chain Optimization",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
st.markdown("""
<style>
    /* Main container and font styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }
    
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
    }
    
    .badge-normal { background-color: #065f46; color: #6ee7b7; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    .badge-low { background-color: #854d0e; color: #fde047; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    .badge-critical { background-color: #991b1b; color: #fca5a5; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    .badge-out { background-color: #450a0a; color: #f87171; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    .badge-near { background-color: #713f12; color: #fdba74; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    .badge-expired { background-color: #7f1d1d; color: #fecaca; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.75rem; }
    
    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #fbbf24;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize DB session and auto-seed on clean deployment
@st.cache_resource
def get_database_engine():
    init_db()
    db = SessionLocal()
    try:
        if db.query(models.Drug).count() == 0:
            from data.generate_dataset import generate_full_dataset, seed_database_from_dataset
            df_d, df_c = generate_full_dataset()
            seed_database_from_dataset(db, df_d, df_c)
    except Exception as e:
        print(f"[INIT] DB Auto-seed notice: {e}")
    finally:
        db.close()
    return True

get_database_engine()

def get_db():
    return SessionLocal()


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pill.png", width=64)
    st.markdown("### **PharmaAI**")
    st.markdown("**Drug Stock & Supply Optimization**")
    st.caption("AI-Powered Decision Support System v1.0.0")
    
    st.divider()
    
    st.markdown("#### **System Diagnostics**")
    db_test = get_db()
    try:
        drug_count = db_test.query(models.Drug).count()
        cons_count = db_test.query(models.Consumption).count()
        st.success(f"Database: Online ({drug_count} Drugs, {cons_count:,} Historical Records)")
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
    finally:
        db_test.close()
        
    st.markdown("#### **Optimization Controls**")
    global_budget = st.number_input("Max Budget ($)", min_value=1000.0, max_value=500000.0, value=50000.0, step=5000.0)
    global_storage = st.number_input("Warehouse Unit Capacity", min_value=5000, max_value=100000, value=35000, step=2500)
    service_level = st.selectbox("Safety Service Level", options=["95% (Z=1.65)", "99% (Z=2.33)", "90% (Z=1.28)"], index=0)
    z_score = 1.65 if "1.65" in service_level else (2.33 if "2.33" in service_level else 1.28)
    
    st.divider()
    
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Administrative System Only</strong><br>
        This platform operates as an inventory logistics decision-support tool. It does NOT offer medical or clinical advice.
    </div>
    """, unsafe_allow_html=True)


# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">
                AI-Based Drug Stock & Supply Chain Optimization
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Autonomous Demand Forecasting (Random Forest) • Dynamic Reorder Points • PuLP Linear Programming Replenishment
            </p>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.08); padding: 10px 16px; border-radius: 12px;">
            <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 700;">HOSPITAL STORE MODE</span><br>
            <span style="font-size: 0.95rem; font-weight: 600;">ACTIVE INVENTORY ENGINE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs Navigation
tabs = st.tabs([
    "📊 TAB 1 — OVERVIEW",
    "📦 TAB 2 — INVENTORY",
    "📈 TAB 3 — DEMAND FORECAST",
    "⚙️ TAB 4 — REORDER OPTIMIZATION",
    "⚠️ TAB 5 — EXPIRY ALERTS",
    "🚚 TAB 6 — SUPPLIERS & ORDERS"
])


# =========================================================================
# TAB 1: OVERVIEW
# =========================================================================
with tabs[0]:
    st.markdown("### **Executive Supply-Chain KPI Dashboard**")
    
    db = get_db()
    try:
        kpis = crud.get_inventory_summary_kpis(db)
        monthly_data = crud.get_monthly_consumption_summary(db)
        top_drugs = crud.get_top_consumed_drugs(db, limit=10)
        low_stock_items, _ = crud.get_inventory_items(db, limit=10, status_filter="LOW STOCK")
        critical_items, _ = crud.get_inventory_items(db, limit=10, status_filter="CRITICAL")
        out_items, _ = crud.get_inventory_items(db, limit=10, status_filter="OUT OF STOCK")
    finally:
        db.close()
        
    # Top KPI Cards Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Drugs</div>
            <div class="kpi-value">{kpis['total_drugs']}</div>
            <div class="kpi-subtext">Active catalog SKUs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Stock Units</div>
            <div class="kpi-value" style="color: #38bdf8;">{kpis['total_stock_units']:,}</div>
            <div class="kpi-subtext">Warehouse inventory</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(245, 158, 11, 0.4);">
            <div class="kpi-title" style="color: #fbbf24;">Low Stock Items</div>
            <div class="kpi-value" style="color: #f59e0b;">{kpis['low_stock_items']}</div>
            <div class="kpi-subtext">At or below reorder pt</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="kpi-title" style="color: #f87171;">Critical Items</div>
            <div class="kpi-value" style="color: #ef4444;">{kpis['critical_items']}</div>
            <div class="kpi-subtext">Below min safety floor</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(249, 115, 22, 0.4);">
            <div class="kpi-title" style="color: #fb923c;">Expiring Soon</div>
            <div class="kpi-value" style="color: #f97316;">{kpis['near_expiry']}</div>
            <div class="kpi-subtext">Within 30 calendar days</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c6:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(220, 38, 38, 0.7);">
            <div class="kpi-title" style="color: #fca5a5;">Out of Stock</div>
            <div class="kpi-value" style="color: #dc2626;">{kpis['out_of_stock']}</div>
            <div class="kpi-subtext">Immediate action needed</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    
    # Visual Analytics Row 1
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### **Inventory Distribution by Drug Category**")
        cat_df = pd.DataFrame(list(kpis["category_stock"].items()), columns=["Category", "Total Units"])
        fig_cat = px.pie(
            cat_df,
            values="Total Units",
            names="Category",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cat.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25)
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### **Monthly Hospital Consumption Trend (Units)**")
        if monthly_data:
            m_df = pd.DataFrame(monthly_data)
            fig_month = px.bar(
                m_df,
                x="month",
                y="quantity",
                text="quantity",
                color="quantity",
                color_continuous_scale="Viridis",
                labels={"month": "Month", "quantity": "Units Consumed"}
            )
            fig_month.update_traces(textposition="outside")
            fig_month.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("No consumption data available yet.")
            
    # Visual Analytics Row 2
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("#### **Top 10 High-Velocity Hospital Medications**")
        if top_drugs:
            top_df = pd.DataFrame(top_drugs)
            fig_top = px.bar(
                top_df,
                x="total_consumed",
                y="drug_name",
                orientation="h",
                color="category",
                labels={"total_consumed": "Total Units Consumed", "drug_name": "Drug Name"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_top.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
    with col_chart4:
        st.markdown("#### **Immediate Stockout Risk Queue**")
        urgent_list = out_items + critical_items + low_stock_items
        if urgent_list:
            u_df = pd.DataFrame(urgent_list)[["drug_name", "category", "current_stock", "reorder_point", "status"]].head(8)
            fig_risk = px.scatter(
                u_df,
                x="reorder_point",
                y="current_stock",
                text="drug_name",
                color="status",
                size=[30] * len(u_df),
                color_discrete_map={
                    "OUT OF STOCK": "#dc2626",
                    "CRITICAL": "#ea580c",
                    "LOW STOCK": "#eab308"
                },
                labels={"reorder_point": "Reorder Threshold Point", "current_stock": "Current Inventory Stock"}
            )
            fig_risk.add_shape(
                type="line", line=dict(dash="dash", color="#94a3b8"),
                x0=0, x1=max(u_df["reorder_point"]), y0=0, y1=max(u_df["reorder_point"])
            )
            fig_risk.update_traces(textposition="top center")
            fig_risk.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.success("All inventory stock levels are healthy.")


# =========================================================================
# TAB 2: INVENTORY
# =========================================================================
with tabs[1]:
    st.markdown("### **Hospital Drug Inventory & Live Stock Monitoring**")
    
    # Filter and Search Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1.5, 1.5, 1])
    
    with f_col1:
        inv_search = st.text_input("🔍 Search Drug Name, Manufacturer, or Batch", placeholder="e.g. Paracetamol, BAT-24...")
    with f_col2:
        status_choice = st.selectbox(
            "Filter by Stock Status",
            options=["ALL", "NORMAL", "LOW STOCK", "CRITICAL", "OUT OF STOCK", "NEAR EXPIRY", "EXPIRED"]
        )
    with f_col3:
        cat_choice = st.selectbox(
            "Filter by Category",
            options=["ALL", "Antibiotics", "Analgesics & Antipyretics", "Cardiovascular", "Antidiabetic", "Respiratory", "Gastrointestinal", "Antiviral & Anti-infective", "Emergency & Critical Care"]
        )
    with f_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        show_add_modal = st.button("➕ Add Drug", use_container_width=True)

    # Add Drug Expander
    if show_add_modal:
        with st.expander("📝 Register New Drug & Initial Inventory", expanded=True):
            with st.form("add_drug_form"):
                af1, af2 = st.columns(2)
                with af1:
                    new_name = st.text_input("Drug Name *", placeholder="e.g. Cefixime 200mg")
                    new_cat = st.selectbox("Category *", options=["Antibiotics", "Analgesics & Antipyretics", "Cardiovascular", "Antidiabetic", "Respiratory", "Gastrointestinal", "Antiviral & Anti-infective", "Emergency & Critical Care"])
                    new_mfg = st.text_input("Manufacturer", placeholder="Apex Pharma")
                    new_batch = st.text_input("Batch Number", placeholder="BAT-882-AZ")
                with af2:
                    new_expiry = st.date_input("Expiry Date *", value=date.today() + timedelta(days=365))
                    new_price = st.number_input("Unit Price ($) *", min_value=0.5, value=15.0, step=1.0)
                    new_stock = st.number_input("Initial Current Stock", min_value=0, value=150, step=10)
                    new_min = st.number_input("Minimum Stock Floor", min_value=5, value=25, step=5)
                    new_max = st.number_input("Maximum Storage Ceiling", min_value=50, value=600, step=50)
                    new_rop = st.number_input("Reorder Point", min_value=10, value=60, step=5)
                    
                submitted = st.form_submit_button("Submit and Save Drug", use_container_width=True)
                if submitted:
                    if not new_name.strip():
                        st.error("Drug Name is required.")
                    else:
                        db = get_db()
                        try:
                            d_in = schemas.DrugCreate(
                                drug_name=new_name.strip(),
                                category=new_cat,
                                manufacturer=new_mfg,
                                batch_number=new_batch,
                                expiry_date=new_expiry,
                                unit_price=float(new_price),
                                initial_stock=int(new_stock),
                                minimum_stock=int(new_min),
                                maximum_stock=int(new_max),
                                reorder_point=int(new_rop)
                            )
                            crud.create_drug(db, d_in)
                            st.success(f"Drug '{new_name}' successfully added to catalog!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error creating drug: {ex}")
                        finally:
                            db.close()

    # Load inventory from database
    db = get_db()
    try:
        inv_list, total_count = crud.get_inventory_items(
            db,
            limit=250,
            status_filter=status_choice if status_choice != "ALL" else None,
            category_filter=cat_choice if cat_choice != "ALL" else None,
            search=inv_search if inv_search.strip() else None
        )
    finally:
        db.close()
        
    if inv_list:
        table_data = []
        for i in inv_list:
            table_data.append({
                "ID": i["drug_id"],
                "Drug Name": i["drug_name"],
                "Category": i["category"],
                "Current Stock": i["current_stock"],
                "Min Stock": i["minimum_stock"],
                "Reorder Point": i["reorder_point"],
                "Unit Price ($)": f"${i['unit_price']:.2f}",
                "Expiry Date": str(i["expiry_date"]),
                "Status": i["status"]
            })
            
        df_table = pd.DataFrame(table_data)
        
        # Color stylized dataframe display
        def color_status(val):
            if val == "NORMAL":
                return "color: #10b981; font-weight: bold;"
            elif val == "LOW STOCK":
                return "color: #f59e0b; font-weight: bold;"
            elif val == "CRITICAL":
                return "color: #ef4444; font-weight: bold;"
            elif val == "OUT OF STOCK":
                return "color: #dc2626; font-weight: bold; background-color: rgba(220,38,38,0.1);"
            elif val == "NEAR EXPIRY":
                return "color: #f97316; font-weight: bold;"
            elif val == "EXPIRED":
                return "color: #b91c1c; font-weight: bold; background-color: rgba(185,28,28,0.15);"
            return ""
            
        st.dataframe(
            df_table.style.map(color_status, subset=["Status"]),
            use_container_width=True,
            height=450
        )
        st.caption(f"Showing {len(df_table)} matching medications.")
        
        # Quick Stock Adjustment Form
        with st.expander("⚡ Quick Stock Adjustment / Replenishment Check-In"):
            with st.form("quick_adj_form"):
                qa1, qa2, qa3 = st.columns(3)
                with qa1:
                    adj_drug_name = st.selectbox("Select Drug", options=[f"{d['ID']}: {d['Drug Name']}" for d in table_data])
                    adj_id = int(adj_drug_name.split(":")[0])
                with qa2:
                    adj_action = st.radio("Action Type", ["Receive Stock (+)", "Dispense / Consume (-)", "Set Absolute Value"])
                with qa3:
                    adj_qty = st.number_input("Quantity Units", min_value=1, value=50, step=10)
                    
                qa_submit = st.form_submit_button("Update Stock Level")
                if qa_submit:
                    db = get_db()
                    try:
                        inv_rec = crud.get_inventory_by_drug_id(db, adj_id)
                        if inv_rec:
                            if adj_action == "Receive Stock (+)":
                                inv_rec.current_stock += int(adj_qty)
                            elif adj_action == "Dispense / Consume (-)":
                                inv_rec.current_stock = max(0, inv_rec.current_stock - int(adj_qty))
                            else:
                                inv_rec.current_stock = int(adj_qty)
                            db.commit()
                            st.success(f"Updated {inv_rec.drug.drug_name} stock to {inv_rec.current_stock} units.")
                            st.rerun()
                    finally:
                        db.close()
    else:
        st.warning("No medications match the chosen criteria.")


# =========================================================================
# TAB 3: DEMAND FORECAST
# =========================================================================
with tabs[2]:
    st.markdown("### **AI-Powered Demand Forecasting (Random Forest)**")
    st.caption("Trained on multi-variate historical consumption lags, day-of-week seasonality, and category trends.")
    
    db = get_db()
    try:
        all_drugs = db.query(models.Drug).order_by(models.Drug.drug_name.asc()).all()
        drug_options = {d.drug_name: d.drug_id for d in all_drugs}
    finally:
        db.close()
        
    fc_col1, fc_col2, fc_col3 = st.columns([2, 1, 1])
    with fc_col1:
        selected_drug_name = st.selectbox("Select Target Drug for Demand Prediction", options=list(drug_options.keys()))
        selected_drug_id = drug_options[selected_drug_name]
    with fc_col2:
        forecast_horizon = st.radio("Forecast Horizon", [7, 30], horizontal=True, index=1)
    with fc_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_fc_btn = st.button("🔮 Generate ML Forecast", use_container_width=True)
        
    # Generate Forecast
    db = get_db()
    try:
        with st.spinner("Executing Random Forest multi-step recursive forecasting..."):
            forecast_result = generate_drug_forecast(
                db,
                drug_id=selected_drug_id,
                days_ahead=forecast_horizon,
                service_level_z=z_score
            )
    except Exception as e:
        st.error(f"Forecasting Engine Error: {e}")
        forecast_result = None
    finally:
        db.close()
        
    if forecast_result:
        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Avg Daily Demand", f"{forecast_result.average_daily_demand} units/day")
        with m2:
            st.metric(f"Total {forecast_horizon}-Day Demand", f"{int(forecast_result.total_predicted_demand)} units")
        with m3:
            st.metric("Calculated Safety Stock", f"{forecast_result.safety_stock_calculated} units")
        with m4:
            st.metric("Dynamic Reorder Point", f"{forecast_result.reorder_point_calculated} units")
        with m5:
            st.metric("Current Stock Level", f"{forecast_result.current_stock} units", 
                      delta="At/Below ROP" if forecast_result.current_stock <= forecast_result.reorder_point_calculated else "Healthy",
                      delta_color="inverse" if forecast_result.current_stock <= forecast_result.reorder_point_calculated else "normal")
            
        # Model Performance Metrics Banner
        metrics = forecast_result.metrics or {}
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 20px; border-radius: 10px; margin: 16px 0; border: 1px solid rgba(255,255,255,0.06); display: flex; gap: 30px; font-size: 0.88rem;">
            <span><strong>Model Architecture:</strong> Random Forest Regressor</span>
            <span><strong>MAE:</strong> <span style="color: #38bdf8;">{metrics.get('mae', 'N/A')} units</span></span>
            <span><strong>RMSE:</strong> <span style="color: #a855f7;">{metrics.get('rmse', 'N/A')} units</span></span>
            <span><strong>R² Score:</strong> <span style="color: #10b981;">{metrics.get('r2', 'N/A')}</span></span>
            <span><strong>Status Classification:</strong> <span style="color: #f59e0b;">{forecast_result.status}</span></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Plotly Time Series Graph
        hist_df = pd.DataFrame(forecast_result.historical_points)
        pred_df = pd.DataFrame([p.model_dump() for p in forecast_result.forecast_points])
        
        fig_fc = go.Figure()
        
        # Historical actual consumption
        if not hist_df.empty:
            fig_fc.add_trace(go.Scatter(
                x=hist_df["date"],
                y=hist_df["actual_demand"],
                name="Historical Actual Consumption",
                line=dict(color="#38bdf8", width=2.5),
                mode="lines+markers"
            ))
            
        # Forecasted Demand
        if not pred_df.empty:
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["predicted_demand"],
                name=f"Predicted Demand (Next {forecast_horizon} Days)",
                line=dict(color="#10b981", width=3, dash="solid"),
                mode="lines+markers"
            ))
            
            # Confidence Interval Upper Bound
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["confidence_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            
            # Confidence Interval Lower Bound with fill
            fig_fc.add_trace(go.Scatter(
                x=pred_df["date"],
                y=pred_df["confidence_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.15)",
                name=f"Confidence Band ({service_level.split()[0]})"
            ))
            
        # Add Reorder Threshold Line
        fig_fc.add_hline(
            y=forecast_result.reorder_point_calculated,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Reorder Point: {forecast_result.reorder_point_calculated}",
            annotation_position="bottom right"
        )
        
        fig_fc.update_layout(
            title=f"Demand Trajectory & Multi-Step Forecast for {selected_drug_name}",
            xaxis_title="Timeline",
            yaxis_title="Units Daily",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)


# =========================================================================
# TAB 4: REORDER OPTIMIZATION (PuLP LINEAR PROGRAMMING)
# =========================================================================
with tabs[3]:
    st.markdown("### **Linear Programming Reorder Optimization (PuLP Engine)**")
    st.caption("Mathematically optimizes replenishment batches considering current stock, safety buffers, lead time, warehouse ceiling, and procurement budget.")
    
    opt_col1, opt_col2, opt_col3 = st.columns([1.5, 1.5, 1])
    with opt_col1:
        budget_input = st.number_input("Procurement Budget Limit ($)", min_value=1000.0, max_value=500000.0, value=float(global_budget), step=5000.0)
    with opt_col2:
        storage_input = st.number_input("Maximum Storage Capacity (Units)", min_value=5000, max_value=100000, value=int(global_storage), step=2500)
    with opt_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        gen_btn = st.button("🚀 Generate Reorder Recommendation", use_container_width=True)
        
    db = get_db()
    try:
        with st.spinner("Solving Mixed-Integer Linear Program with PuLP..."):
            req_opt = schemas.OptimizationRequest(
                budget_limit=budget_input,
                storage_limit=storage_input,
                service_level_z=z_score
            )
            opt_res = optimize_reorder_quantities(db, req_opt)
    except Exception as e:
        st.error(f"Optimization Solver Error: {e}")
        opt_res = None
    finally:
        db.close()
        
    if opt_res:
        # Summary Row
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Replenishment Cost", f"${opt_res.total_estimated_cost:,.2f}",
                      delta=f"{opt_res.budget_utilized_percent}% Budget Used")
        with s2:
            st.metric("Units to Order", f"{opt_res.total_units_ordered:,} units")
        with s3:
            st.metric("Drugs Requiring Reorder", f"{opt_res.drugs_to_reorder_count} of {opt_res.total_drugs_analyzed}")
        with s4:
            st.metric("PuLP Solver Status", f"{opt_res.optimization_status}", delta="Optimal Solution Found")
            
        st.markdown("#### **Optimized Order Plan Table**")
        
        # Build table
        rec_data = []
        for r in opt_res.recommendations:
            rec_data.append({
                "Drug ID": r.drug_id,
                "Drug Name": r.drug_name,
                "Category": r.category,
                "Current Stock": r.current_stock,
                "Reorder Point": r.reorder_point,
                "Safety Stock": r.safety_stock,
                "Predicted Lead Demand": r.predicted_demand,
                "Recommended Quantity": r.recommended_order_quantity,
                "Expected Stock After": r.expected_inventory_after_order,
                "Unit Price": f"${r.unit_price:.2f}",
                "Estimated Cost": f"${r.estimated_cost:,.2f}",
                "Status": r.stock_status,
                "Supplier": r.supplier_name
            })
            
        df_opt = pd.DataFrame(rec_data)
        
        # Color highlighting for items that need to be reordered
        def highlight_orders(row):
            if row["Recommended Quantity"] > 0:
                return ["background-color: rgba(14, 165, 233, 0.12); font-weight: 600;"] * len(row)
            return [""] * len(row)
            
        st.dataframe(df_opt.style.apply(highlight_orders, axis=1), use_container_width=True, height=420)
        
        # Instant Purchase Order Creation from Recommendation
        st.markdown("#### **Instant Purchase Order Dispatch**")
        with st.expander("📝 Generate Automated Purchase Order from Optimization Plan"):
            active_reorders = [r for r in opt_res.recommendations if r.recommended_order_quantity > 0]
            if active_reorders:
                po_choices = [f"{r.drug_id}: {r.drug_name} | Qty: {r.recommended_order_quantity} | Cost: ${r.estimated_cost:,.2f} | Supplier: {r.supplier_name}" for r in active_reorders]
                selected_po_item = st.selectbox("Select Recommendation to Create PO", options=po_choices)
                sel_id = int(selected_po_item.split(":")[0])
                sel_rec = next(r for r in active_reorders if r.drug_id == sel_id)
                
                if st.button("Confirm and Place Purchase Order", use_container_width=True):
                    db = get_db()
                    try:
                        sup_id = sel_rec.supplier_id
                        if not sup_id:
                            first_sup = db.query(models.Supplier).first()
                            sup_id = first_sup.supplier_id if first_sup else 1
                            
                        p_in = schemas.PurchaseCreate(
                            drug_id=sel_rec.drug_id,
                            supplier_id=sup_id,
                            quantity=sel_rec.recommended_order_quantity,
                            purchase_date=date.today(),
                            status="Pending"
                        )
                        crud.create_purchase(db, p_in)
                        st.success(f"Successfully placed Purchase Order for {sel_rec.recommended_order_quantity} units of {sel_rec.drug_name}!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error creating purchase order: {ex}")
                    finally:
                        db.close()
            else:
                st.info("No items currently require replenishment.")


# =========================================================================
# TAB 5: EXPIRY ALERTS
# =========================================================================
with tabs[4]:
    st.markdown("### **Drug Expiry Management & Risk Alerts**")
    st.caption("Active monitoring of expired and nearing-expiration pharmaceuticals to prevent loss and compliance risks.")
    
    db = get_db()
    try:
        expiry_buckets = crud.get_expiry_buckets(db)
    finally:
        db.close()
        
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(220, 38, 38, 0.7);">
            <div class="kpi-title" style="color: #f87171;">Expired Drugs</div>
            <div class="kpi-value" style="color: #ef4444;">{len(expiry_buckets['expired'])}</div>
            <div class="kpi-subtext">Immediate disposal required</div>
        </div>
        """, unsafe_allow_html=True)
    with e_col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(249, 115, 22, 0.5);">
            <div class="kpi-title" style="color: #fb923c;">Expiring in 30 Days</div>
            <div class="kpi-value" style="color: #f97316;">{len(expiry_buckets['within_30_days'])}</div>
            <div class="kpi-subtext">High urgency dispense</div>
        </div>
        """, unsafe_allow_html=True)
    with e_col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(234, 179, 8, 0.4);">
            <div class="kpi-title" style="color: #facc15;">Expiring in 60 Days</div>
            <div class="kpi-value" style="color: #eab308;">{len(expiry_buckets['within_60_days'])}</div>
            <div class="kpi-subtext">Priority stock rotation</div>
        </div>
        """, unsafe_allow_html=True)
    with e_col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(56, 189, 248, 0.4);">
            <div class="kpi-title" style="color: #7dd3fc;">Expiring in 90 Days</div>
            <div class="kpi-value" style="color: #38bdf8;">{len(expiry_buckets['within_90_days'])}</div>
            <div class="kpi-subtext">Standard advance alert</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        f"🚨 Expired ({len(expiry_buckets['expired'])})",
        f"⚡ Within 30 Days ({len(expiry_buckets['within_30_days'])})",
        f"⚠️ Within 60 Days ({len(expiry_buckets['within_60_days'])})",
        f"📅 Within 90 Days ({len(expiry_buckets['within_90_days'])})"
    ])
    
    def render_expiry_table(item_list, alert_label):
        if not item_list:
            st.success(f"No medications found in '{alert_label}'.")
            return
        df_e = pd.DataFrame(item_list)[[
            "drug_name", "category", "batch_number", "expiry_date", "days_until_expiry", "current_stock", "unit_price", "loss_value"
        ]].rename(columns={
            "drug_name": "Drug Name",
            "category": "Category",
            "batch_number": "Batch",
            "expiry_date": "Expiry Date",
            "days_until_expiry": "Days Left",
            "current_stock": "Current Stock",
            "unit_price": "Unit Price ($)",
            "loss_value": "Estimated Value ($)"
        })
        st.dataframe(df_e, use_container_width=True)
        total_loss = df_e["Estimated Value ($)"].sum()
        st.caption(f"Total inventory value in this category: **${total_loss:,.2f}**")
        
    with subtab1:
        render_expiry_table(expiry_buckets["expired"], "Expired")
    with subtab2:
        render_expiry_table(expiry_buckets["within_30_days"], "Within 30 Days")
    with subtab3:
        render_expiry_table(expiry_buckets["within_60_days"], "Within 60 Days")
    with subtab4:
        render_expiry_table(expiry_buckets["within_90_days"], "Within 90 Days")


# =========================================================================
# TAB 6: SUPPLIERS & ORDERS
# =========================================================================
with tabs[5]:
    st.markdown("### **Supplier Management & Purchase Order Tracking**")
    
    sup_col1, sup_col2 = st.columns([1, 2])
    
    db = get_db()
    try:
        suppliers = crud.get_suppliers(db)
        purchases = crud.get_purchases(db)
    finally:
        db.close()
        
    with sup_col1:
        st.markdown("#### **Registered Pharmaceutical Vendors**")
        for s in suppliers:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 14px 18px; border-radius: 12px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">{s.supplier_name}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0;">📞 {s.contact or 'N/A'}</div>
                <div style="font-size: 0.82rem; color: #38bdf8; font-weight: 600;">⏱️ Lead Time: {s.lead_time_days} days</div>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("➕ Register New Supplier"):
            with st.form("new_supplier_form"):
                ns_name = st.text_input("Supplier Name *")
                ns_contact = st.text_input("Contact Info (Phone / Email)")
                ns_lead = st.number_input("Delivery Lead Time (Days)", min_value=1, max_value=60, value=7)
                ns_submit = st.form_submit_button("Add Supplier")
                if ns_submit:
                    if not ns_name.strip():
                        st.error("Supplier Name is required.")
                    else:
                        db = get_db()
                        try:
                            s_in = schemas.SupplierCreate(
                                supplier_name=ns_name.strip(),
                                contact=ns_contact.strip() if ns_contact else None,
                                lead_time_days=int(ns_lead)
                            )
                            crud.create_supplier(db, s_in)
                            st.success(f"Supplier '{ns_name}' successfully added!")
                            st.rerun()
                        finally:
                            db.close()
                            
    with sup_col2:
        st.markdown("#### **Procurement Purchase Orders**")
        if purchases:
            df_p = pd.DataFrame(purchases)[[
                "purchase_id", "drug_name", "supplier_name", "quantity", "unit_price", "total_cost", "purchase_date", "expected_delivery_date", "status"
            ]].rename(columns={
                "purchase_id": "PO #",
                "drug_name": "Drug Name",
                "supplier_name": "Supplier",
                "quantity": "Quantity",
                "unit_price": "Unit Price ($)",
                "total_cost": "Total Cost ($)",
                "purchase_date": "Order Date",
                "expected_delivery_date": "Expected Delivery",
                "status": "Status"
            })
            
            def color_po_status(val):
                if val == "Delivered":
                    return "color: #10b981; font-weight: bold;"
                elif val == "Pending":
                    return "color: #f59e0b; font-weight: bold;"
                return "color: #ef4444; font-weight: bold;"
                
            st.dataframe(df_p.style.map(color_po_status, subset=["Status"]), use_container_width=True, height=360)
            
            # Action: Check-in / Receive Order
            with st.expander("📦 Check-in Delivered Purchase Order (Increments Stock)"):
                pending_orders = [p for p in purchases if p["status"] == "Pending"]
                if pending_orders:
                    po_pick = st.selectbox(
                        "Select Order to Mark Delivered",
                        options=[f"PO #{p['purchase_id']}: {p['drug_name']} ({p['quantity']} units from {p['supplier_name']})" for p in pending_orders]
                    )
                    picked_po_id = int(po_pick.split(":")[0].replace("PO #", "").strip())
                    if st.button("Mark Order as Delivered & Add Stock", use_container_width=True):
                        db = get_db()
                        try:
                            crud.update_purchase_status(db, picked_po_id, "Delivered")
                            st.success(f"PO #{picked_po_id} checked-in! Inventory current_stock successfully updated.")
                            st.rerun()
                        finally:
                            db.close()
                else:
                    st.info("No pending purchase orders to receive.")
        else:
            st.info("No purchase orders on record.")
