# 💊 AI-Based Drug Stock & Supply Chain Optimization System

> **A production-ready, AI-driven inventory intelligence and replenishment platform designed for hospitals, healthcare facilities, and regional public-health drug stores.**

---

## ⚠️ Clinical & Administrative Disclaimer
> **IMPORTANT:** This software is an administrative logistics and supply-chain decision-support system. It does **NOT** provide medical diagnosis, drug therapy recommendations, dosage calculations, or clinical healthcare advice. All inventory decisions must be reviewed by qualified pharmacy and supply chain personnel.

---

## 1. Project Overview

In healthcare institutions, drug inventory management carries life-or-death implications. Overstocking leads to financial loss through medication expiry, while understocking risks catastrophic stockouts of critical antibiotics, analgesics, and emergency cardiovascular medicines.

The **AI-Based Drug Stock & Supply Chain Optimization System** solves this challenge through:
1. **Dynamic Stock Classification**: Continuous monitoring of drug thresholds into `NORMAL`, `LOW STOCK`, `CRITICAL`, `OUT OF STOCK`, `NEAR EXPIRY`, and `EXPIRED`.
2. **Autonomous Demand Forecasting**: Multi-step recursive regression powered by **Random Forest** that learns calendar seasonality, weekday-weekend hospital admission swings, and historical consumption lags.
3. **Dynamic Reorder Point & Safety Stock**: Formulated through lead-time demand variability and target service levels ($Z$-scores).
4. **Constrained Linear Programming Replenishment**: Powered by **PuLP (Mixed-Integer Linear Programming)** to solve the global multi-item reorder allocation problem under procurement budget and physical warehouse storage limits.
5. **Interactive Operations Dashboard**: High-aesthetic **Streamlit + Plotly** control room featuring 6 functional tabs.

---

## 2. Key Features

- **Inventory Management**:
  - Add, update, view, and delete drugs with catalog metadata (manufacturer, batch, expiry, unit price).
  - Searchable and filterable live stock table.
  - One-click stock adjustments and receipt check-ins.

- **Stock Monitoring & Classifications**:
  - `NORMAL`: Stock > Reorder Point.
  - `LOW STOCK`: Stock $\le$ Reorder Point.
  - `CRITICAL`: Stock $\le$ Minimum Safety Floor.
  - `OUT OF STOCK`: Stock $= 0$.
  - `NEAR EXPIRY`: Expiry within 30 calendar days.
  - `EXPIRED`: Passed expiration date.

- **Demand Forecasting (Machine Learning)**:
  - Multi-step ahead predictions (7 or 30 days).
  - Autoregressive lags (Lag 1, 7, 14, 30) + rolling statistical windows (7, 14, 30 days).
  - Confidence interval estimation using Random Forest tree ensemble variance.
  - Model evaluation with **MAE**, **RMSE**, **R²**, and **MAPE**.

- **Reorder Point (ROP) & Safety Stock Calculation**:
  $$\text{Safety Stock} (SS) = Z \times \sigma_{\text{daily}} \times \sqrt{L}$$
  $$\text{Reorder Point} (ROP) = (\mu_{\text{daily}} \times L) + SS$$
  *(where $L$ = supplier lead time in days, $\mu$ = mean daily demand, $\sigma$ = daily demand standard deviation, $Z$ = service level factor).*

- **Linear Programming Reorder Quantity Optimization**:
  - PuLP mathematical solver computing optimal order quantities $Q_i$.
  - Constrained by total procurement budget (\$) and warehouse volume ceilings (units).
  - Prioritizes life-critical and out-of-stock items in constrained budget regimes.

- **Expiry Management**:
  - Segmented risk buckets: Expired, $<30$ days, $<60$ days, $<90$ days.
  - Quantifies total dollar loss on nearing-expiry and expired batches.

- **Supplier Directory & Purchase Order Tracking**:
  - Vendor directory with lead times and contact channels.
  - Purchase order life-cycle tracking (`Pending` $\to$ `Delivered` $\to$ `Cancelled`).
  - Automated inventory current-stock increment upon delivery check-in.

---

## 3. System Architecture

```mermaid
graph TD
    A[Hospital Operators / Store Managers] -->|Streamlit Dashboard :8501| B(Frontend Control Room)
    B -->|REST API Calls| C(FastAPI Backend :8000)
    
    subgraph Data & Storage Layer
        D[(MySQL / SQLite Database)]
        E[data/drug_inventory.csv >=10k Records]
    end

    subgraph Intelligence & Optimization Layer
        F[ml/train_model.py - Random Forest Regressor]
        G[ml/model.pkl Serialized Model Bundle]
        H[Forecasting Engine - Multi-Step Recursive]
        I[PuLP Linear Programming Solver - CBC]
    end

    C <-->|SQLAlchemy ORM Transactions| D
    E -->|Feature Engineering & Lag Extraction| F
    F -->|Joblib Serialization| G
    G -->|Warm-up & Inference| H
    H -->|Forecasts & Dynamic ROP| C
    I -->|Cost-Optimal Reorder Quantities| C
```

---

## 4. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend REST API** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **Database ORM** | SQLAlchemy 2.0, PyMySQL, Cryptography, SQLite3 |
| **Machine Learning** | Scikit-learn (RandomForestRegressor), Pandas, NumPy, Joblib |
| **Optimization** | PuLP 2.9 (Mixed-Integer Linear Programming with CBC Solver) |
| **Frontend UI** | Streamlit, Plotly Express & Graph Objects |
| **Quality & Tests** | Pytest, TestClient, Flake8 |

---

## 5. Database Schema & Setup

The database schema is fully compatible with **MySQL Workbench**, **phpMyAdmin**, and local MySQL/MariaDB instances. A zero-config **SQLite3** option is also provided for instant local demonstration.

### Tables
1. `suppliers`: `supplier_id`, `supplier_name`, `contact`, `lead_time_days`, `created_at`
2. `drugs`: `drug_id`, `drug_name`, `category`, `manufacturer`, `batch_number`, `expiry_date`, `unit_price`, `created_at`
3. `inventory`: `inventory_id`, `drug_id` (FK), `current_stock`, `minimum_stock`, `maximum_stock`, `reorder_point`, `last_updated`
4. `consumption`: `consumption_id`, `drug_id` (FK), `date`, `quantity_consumed`, `created_at`
5. `purchases`: `purchase_id`, `drug_id` (FK), `supplier_id` (FK), `quantity`, `purchase_date`, `expected_delivery_date`, `status`, `created_at`
6. `forecasts`: `forecast_id`, `drug_id` (FK), `forecast_date`, `predicted_demand`, `confidence_level`, `created_at`

### Setting Up with MySQL:
1. Open **MySQL Workbench** or **phpMyAdmin**.
2. Run the provided [`schema.sql`](file:///schema.sql) file.
3. Configure your MySQL credentials in `.env`:
   ```bash
   DATABASE_URL=mysql+pymysql://root:yourpassword@localhost:3306/drug_stock_db
   ```

---

## 6. Environment Variables (`.env`)

Copy the template from [`.env.example`](file:///c:/Users/91735/Downloads/ai%20based%20drug%20stock/.env.example) to `.env`:

```ini
# Database Connection (Switch between SQLite or MySQL)
DATABASE_URL=sqlite:///./data/drug_stock.db
# DATABASE_URL=mysql+pymysql://root:password@localhost:3306/drug_stock_db

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

# Optimization & Model Defaults
DEFAULT_BUDGET_LIMIT=50000.0
DEFAULT_STORAGE_LIMIT=35000
DEFAULT_SERVICE_LEVEL_Z=1.65
MODEL_PATH=ml/model.pkl
DATASET_PATH=data/drug_inventory.csv
```

---

## 7. Installation & Quickstart

### Step 1: Clone or Open Project Directory
```bash
cd "drug-stock-optimization"
```

### Step 2: Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate Synthetic Dataset & Initialize Database
Generates **11,440+ historical consumption records** across 52 realistic medications and 5 suppliers:
```bash
python data/generate_dataset.py
```

### Step 5: Train Demand Forecasting Model
Trains the Random Forest model and saves `ml/model.pkl`:
```bash
python ml/train_model.py
```

---

## 8. Running the Application

### Start Backend API:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Start Streamlit Dashboard:
```bash
streamlit run dashboard/app.py
```
- Dashboard URL: [http://localhost:8501](http://localhost:8501)

---

## 9. Running Test Suite
Execute the automated test suite covering API routes, ML pipelines, and PuLP optimization:
```bash
pytest -v
```

---

## 10. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | System health and disclaimer |
| `GET` | `/drugs` | Paginated catalog list with search & category filter |
| `POST` | `/drugs` | Create new drug with initial inventory allocation |
| `GET` | `/drugs/{drug_id}` | Fetch detailed drug metadata |
| `PUT` | `/drugs/{drug_id}` | Update drug attributes |
| `DELETE` | `/drugs/{drug_id}` | Delete drug and cascade inventory records |
| `GET` | `/inventory` | Filterable inventory stock list with computed statuses |
| `GET` | `/inventory/summary` | Executive KPI card counters |
| `GET` | `/inventory/low-stock` | Drugs at or below reorder threshold |
| `GET` | `/inventory/critical` | Drugs below minimum safety floor |
| `GET` | `/inventory/out-of-stock` | Zero-stock drugs |
| `GET` | `/inventory/expiring` | Segmented breakdown (expired, <30d, <60d, <90d) |
| `POST` | `/forecast/{drug_id}` | Generate 7/30-day Random Forest demand prediction |
| `GET` | `/forecast/{drug_id}` | Retrieve cached forecast and metrics |
| `POST` | `/optimize-reorder` | PuLP linear programming multi-item reorder solver |
| `POST` | `/optimize-reorder/{drug_id}` | Single-item reorder calculation |
| `GET` | `/suppliers` | List registered suppliers and lead times |
| `POST` | `/suppliers` | Register new supplier |
| `GET` | `/purchases` | List purchase orders with status filtering |
| `POST` | `/purchases` | Create procurement purchase order |
| `PUT` | `/purchases/{id}/status` | Update PO status (auto-increments stock on Delivered) |

---

## 11. Mathematical Formulations

### A. Random Forest Demand Forecasting
Demand at time $t$ for drug $i$ is modeled as a non-linear regression function:
$$\hat{D}_{i,t} = f\left(\text{Category}_i, \text{Month}_t, \text{DayOfWeek}_t, D_{i, t-1}, D_{i, t-7}, D_{i, t-14}, \mu_{7}(t), \mu_{14}(t), \mu_{30}(t)\right)$$
Random Forest aggregates $M=120$ decision trees with bootstrap sampling and feature bagging:
$$\hat{D}(X) = \frac{1}{M}\sum_{m=1}^{M} T_m(X)$$
Confidence bounds are derived using ensemble estimator variance:
$$\sigma_{\text{trees}}^2 = \frac{1}{M}\sum_{m=1}^{M}\left(T_m(X) - \hat{D}(X)\right)^2$$
$$\text{Interval} = \hat{D}(X) \pm Z \cdot \sigma_{\text{trees}}$$

### B. PuLP Linear Programming Formulation
For $N$ candidate medications:
- **Decision Variable**: $Q_i \in \mathbb{Z}_{\ge 0}$ (order quantity for drug $i$)
- **Shortfall Variable**: $U_i \ge 0$ (unmet buffer if budget is constrained)

$$\min \sum_{i=1}^{N} \left( W_i \cdot U_i + 0.001 \cdot C_i \cdot Q_i \right)$$

**Subject to:**
1. **Budget Constraint**:
   $$\sum_{i=1}^{N} C_i \cdot Q_i \le \text{BudgetLimit}$$
2. **Storage Capacity Constraint**:
   $$\sum_{i=1}^{N} Q_i \le \text{StorageLimit} - \sum_{i=1}^{N} S_i$$
3. **Inventory Service Target**:
   $$S_i + Q_i + U_i \ge \text{TargetStock}_i \quad \forall i$$
4. **Physical Drug Storage Cap**:
   $$S_i + Q_i \le \text{MaxStock}_i \quad \forall i$$
5. **Non-negativity & Integrality**:
   $$Q_i \in \{0, 1, 2, \dots\}, \quad U_i \ge 0$$

*(where $W_i$ represents priority weights: 1000 for Out of Stock, 500 for Critical, 200 for Low Stock; $C_i$ is unit purchase price; $S_i$ is current stock).*

---

## 12. Dashboard Tour

1. **TAB 1 — OVERVIEW**: Executive KPI metrics, category stock donut, monthly volume bars, top 10 hospital medicines, and live stockout risk plot.
2. **TAB 2 — INVENTORY**: Real-time stock list with color status badges, category/status filters, keyword search, "Add Drug" modal, and quick stock check-in.
3. **TAB 3 — DEMAND FORECAST**: Target drug selector, 7- or 30-day forecast horizon toggle, interactive Plotly time series with 95% confidence bounds, and MAE/RMSE/R² metrics.
4. **TAB 4 — REORDER OPTIMIZATION**: Budget and storage parameter inputs, "Generate Reorder Recommendation" solver button, detailed order plan, and 1-click Purchase Order dispatch.
5. **TAB 5 — EXPIRY ALERTS**: Four dedicated tabs for expired, <30-day, <60-day, and <90-day medicines, with quantified financial value at risk.
6. **TAB 6 — SUPPLIERS & ORDERS**: Registered vendor cards, supplier creation form, purchase orders tracking, and "Receive Stock" check-in button.

---

## 13. Future Roadmap

- **Multi-Store Warehouse Distribution**: Inter-facility transfer optimization between central hospital repositories and peripheral clinics.
- **Deep Learning Time Series**: Transformer and LSTM architectures for long-range epidemic spike predictions.
- **Automated EDI Supplier Dispatch**: Direct integration with pharmaceutical vendor EDI (Electronic Data Interchange) and ERP protocols.
- **Batch Barcode & QR Scanning**: Mobile camera/handheld scanner integration for automated intake and dispensing.
