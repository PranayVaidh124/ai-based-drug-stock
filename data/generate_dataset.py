"""
Realistic Synthetic Healthcare Drug Inventory & Historical Consumption Dataset Generator
Generates >= 10,000 consumption records for 50+ diverse drugs across 8 categories,
with realistic seasonality, weekend dips, variance, expiry dates, and supplier lead times.
"""

import os
import sys
import random
import math
from datetime import date, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Set deterministic seed for reproducibility
np.random.seed(42)
random.seed(42)

CATEGORIES_CONFIG = {
    "Antibiotics": {
        "season_peak_months": [11, 12, 1, 2, 7, 8],  # Winter & monsoon spikes
        "base_range": (15, 45),
        "price_range": (8.50, 65.00),
        "drugs": [
            ("Amoxicillin 500mg", "Apex Pharma", "CAPSULES"),
            ("Azithromycin 250mg", "BioHealth Labs", "TABLETS"),
            ("Ciprofloxacin 500mg", "NovaCare", "TABLETS"),
            ("Ceftriaxone 1g Inj", "MediSupply Global", "VIALS"),
            ("Doxycycline 100mg", "Apex Pharma", "CAPSULES"),
            ("Metronidazole 400mg", "Zenith Lifesciences", "TABLETS"),
            ("Levofloxacin 500mg", "BioHealth Labs", "TABLETS"),
            ("Amoxicillin-Clavulanate 625mg", "NovaCare", "TABLETS"),
        ]
    },
    "Analgesics & Antipyretics": {
        "season_peak_months": [6, 7, 8, 12, 1],
        "base_range": (35, 90),
        "price_range": (3.00, 22.00),
        "drugs": [
            ("Paracetamol 500mg", "Zenith Lifesciences", "TABLETS"),
            ("Ibuprofen 400mg", "Apex Pharma", "TABLETS"),
            ("Tramadol 50mg", "MediSupply Global", "CAPSULES"),
            ("Diclofenac 50mg", "NovaCare", "TABLETS"),
            ("Aspirin 75mg Gastro-resistant", "BioHealth Labs", "TABLETS"),
            ("Naproxen 250mg", "Apex Pharma", "TABLETS"),
            ("Ketorolac 10mg Inj", "MediSupply Global", "AMPOULES"),
        ]
    },
    "Cardiovascular": {
        "season_peak_months": [12, 1, 2],  # Slightly higher in extreme cold
        "base_range": (25, 60),
        "price_range": (12.00, 78.00),
        "drugs": [
            ("Atorvastatin 20mg", "NovaCare", "TABLETS"),
            ("Amlodipine 5mg", "Zenith Lifesciences", "TABLETS"),
            ("Losartan Potassium 50mg", "Apex Pharma", "TABLETS"),
            ("Metoprolol Tartrate 25mg", "BioHealth Labs", "TABLETS"),
            ("Enalapril 10mg", "NovaCare", "TABLETS"),
            ("Hydrochlorothiazide 25mg", "Zenith Lifesciences", "TABLETS"),
            ("Clopidogrel 75mg", "MediSupply Global", "TABLETS"),
            ("Rosuvastatin 10mg", "Apex Pharma", "TABLETS"),
        ]
    },
    "Antidiabetic": {
        "season_peak_months": [],  # Steady year-round chronic consumption
        "base_range": (30, 70),
        "price_range": (9.00, 110.00),
        "drugs": [
            ("Metformin 500mg ER", "Apex Pharma", "TABLETS"),
            ("Glimepiride 2mg", "BioHealth Labs", "TABLETS"),
            ("Insulin Glargine 100IU/ml", "MediSupply Global", "PENS"),
            ("Sitagliptin 100mg", "NovaCare", "TABLETS"),
            ("Empagliflozin 10mg", "Zenith Lifesciences", "TABLETS"),
            ("Gliclazide 80mg", "Apex Pharma", "TABLETS"),
        ]
    },
    "Respiratory": {
        "season_peak_months": [10, 11, 12, 1, 2],  # Heavy winter / pollution smog spike
        "base_range": (20, 55),
        "price_range": (15.00, 85.00),
        "drugs": [
            ("Salbutamol 100mcg Inhaler", "MediSupply Global", "INHALERS"),
            ("Budesonide 200mcg Turbuhaler", "NovaCare", "INHALERS"),
            ("Montelukast 10mg", "BioHealth Labs", "TABLETS"),
            ("Cetirizine 10mg", "Zenith Lifesciences", "TABLETS"),
            ("Ipratropium Bromide Respules", "Apex Pharma", "RESPULES"),
            ("Fluticasone Furoate Nasal Spray", "NovaCare", "SPRAYS"),
        ]
    },
    "Gastrointestinal": {
        "season_peak_months": [5, 6, 7, 8],  # Summer / monsoon foodborne bugs
        "base_range": (25, 65),
        "price_range": (6.00, 45.00),
        "drugs": [
            ("Omeprazole 20mg", "Apex Pharma", "CAPSULES"),
            ("Pantoprazole 40mg", "Zenith Lifesciences", "TABLETS"),
            ("Ondansetron 4mg", "BioHealth Labs", "TABLETS"),
            ("Domperidone 10mg", "NovaCare", "TABLETS"),
            ("Sucralfate 1g Suspension", "MediSupply Global", "BOTTLES"),
            ("Lactulose 10g/15ml Solution", "Apex Pharma", "BOTTLES"),
        ]
    },
    "Antiviral & Anti-infective": {
        "season_peak_months": [11, 12, 1, 2],
        "base_range": (10, 35),
        "price_range": (25.00, 150.00),
        "drugs": [
            ("Oseltamivir 75mg", "MediSupply Global", "CAPSULES"),
            ("Acyclovir 400mg", "BioHealth Labs", "TABLETS"),
            ("Tenofovir Disoproxil 300mg", "NovaCare", "TABLETS"),
            ("Fluconazole 150mg", "Apex Pharma", "TABLETS"),
            ("Valacyclovir 500mg", "Zenith Lifesciences", "TABLETS"),
        ]
    },
    "Emergency & Critical Care": {
        "season_peak_months": [],
        "base_range": (8, 25),
        "price_range": (18.00, 120.00),
        "drugs": [
            ("Epinephrine 1mg/ml Inj", "MediSupply Global", "AMPOULES"),
            ("Atropine Sulfate 0.6mg Inj", "MediSupply Global", "AMPOULES"),
            ("Naloxone 0.4mg/ml Inj", "Apex Pharma", "AMPOULES"),
            ("Furosemide 20mg/2ml Inj", "BioHealth Labs", "AMPOULES"),
            ("Hydrocortisone 100mg Inj", "Zenith Lifesciences", "VIALS"),
            ("Tranexamic Acid 500mg Inj", "NovaCare", "AMPOULES"),
        ]
    }
}

SUPPLIERS = [
    {"supplier_name": "Apex Pharmaceuticals Ltd", "contact": "+1 (800) 555-0191 / orders@apexpharma.com", "lead_time_days": 5},
    {"supplier_name": "MediSupply Global Inc", "contact": "+1 (800) 555-0144 / support@medisupply.com", "lead_time_days": 7},
    {"supplier_name": "BioHealth Laboratories", "contact": "+1 (800) 555-0178 / logistics@biohealth.org", "lead_time_days": 10},
    {"supplier_name": "NovaCare Logistics Corp", "contact": "+1 (800) 555-0122 / supply@novacare.com", "lead_time_days": 4},
    {"supplier_name": "Zenith Lifesciences", "contact": "+1 (800) 555-0165 / sales@zenithlife.com", "lead_time_days": 8},
]


def generate_full_dataset(output_csv: str = "data/drug_inventory.csv", days_history: int = 220):
    """
    Generates historical daily consumption and catalog metadata.
    220 days * 50 drugs = 11,000+ records (satisfies >= 10,000 requirement).
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    today = date.today()
    start_date = today - timedelta(days=days_history)

    drug_records = []
    consumption_records = []
    drug_id_counter = 1

    # Specific stock profiles to guarantee realistic mix of NORMAL, LOW STOCK, CRITICAL, OUT OF STOCK, EXPIRED, NEAR EXPIRY
    status_assignments = [
        "NORMAL", "NORMAL", "LOW_STOCK", "CRITICAL", "OUT_OF_STOCK",
        "NEAR_EXPIRY", "EXPIRED", "NORMAL", "LOW_STOCK", "NORMAL"
    ]

    for cat_name, cat_data in CATEGORIES_CONFIG.items():
        season_peaks = set(cat_data["season_peak_months"])
        b_min, b_max = cat_data["base_range"]
        p_min, p_max = cat_data["price_range"]

        for drug_name, mfg, dosage_form in cat_data["drugs"]:
            drug_id = drug_id_counter
            drug_id_counter += 1

            unit_price = round(random.uniform(p_min, p_max), 2)
            batch_num = f"BAT-{random.randint(100, 999)}-{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"

            # Assign designated profile
            profile = status_assignments[(drug_id - 1) % len(status_assignments)]

            # Determine expiry date
            if profile == "EXPIRED":
                expiry = today - timedelta(days=random.randint(5, 60))
            elif profile == "NEAR_EXPIRY":
                expiry = today + timedelta(days=random.randint(5, 25))
            elif drug_id % 7 == 0:
                expiry = today + timedelta(days=random.randint(35, 55))  # within 60 days
            elif drug_id % 9 == 0:
                expiry = today + timedelta(days=random.randint(65, 85))  # within 90 days
            else:
                expiry = today + timedelta(days=random.randint(180, 720))

            # Daily base consumption rate
            drug_base_demand = random.uniform(b_min, b_max)
            std_dev = drug_base_demand * random.uniform(0.15, 0.30)

            # Lead time calculation
            supplier_idx = random.randint(0, len(SUPPLIERS) - 1)
            lead_time = SUPPLIERS[supplier_idx]["lead_time_days"]

            # Safety Stock: Z * std_dev * sqrt(L)
            safety_stock = int(math.ceil(1.65 * std_dev * math.sqrt(lead_time)))
            reorder_point = int(math.ceil(drug_base_demand * lead_time + safety_stock))
            minimum_stock = max(10, int(safety_stock * 0.7))
            maximum_stock = int(reorder_point * 2.5 + safety_stock)

            # Determine initial stock according to profile
            if profile == "OUT_OF_STOCK":
                current_stock = 0
            elif profile == "CRITICAL":
                current_stock = random.randint(1, minimum_stock)
            elif profile == "LOW_STOCK":
                current_stock = random.randint(minimum_stock + 1, reorder_point)
            else:
                current_stock = random.randint(reorder_point + 10, maximum_stock)

            drug_records.append({
                "drug_id": drug_id,
                "drug_name": drug_name,
                "category": cat_name,
                "manufacturer": mfg,
                "batch_number": batch_num,
                "expiry_date": expiry.strftime("%Y-%m-%d"),
                "unit_price": unit_price,
                "current_stock": current_stock,
                "minimum_stock": minimum_stock,
                "maximum_stock": maximum_stock,
                "reorder_point": reorder_point,
                "supplier_name": SUPPLIERS[supplier_idx]["supplier_name"],
                "lead_time_days": lead_time
            })

            # Generate historical daily consumption
            for d_idx in range(days_history):
                current_date = start_date + timedelta(days=d_idx)
                month = current_date.month
                weekday = current_date.weekday()

                # Multipliers
                season_mult = 1.35 if month in season_peaks else 1.0
                weekend_mult = 0.70 if weekday in (5, 6) else 1.05  # lower on weekends, busy Mon-Fri

                mean_val = drug_base_demand * season_mult * weekend_mult
                # Random demand variation with normal distribution
                daily_consumed = int(max(0, np.random.normal(loc=mean_val, scale=std_dev)))

                consumption_records.append({
                    "drug_id": drug_id,
                    "drug_name": drug_name,
                    "category": cat_name,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "quantity_consumed": daily_consumed,
                    "unit_price": unit_price
                })

    df_drugs = pd.DataFrame(drug_records)
    df_consumption = pd.DataFrame(consumption_records)

    # Save consumption dataset
    df_consumption.to_csv(output_csv, index=False)
    print(f"Generated {len(df_consumption)} consumption records across {len(df_drugs)} drugs in {output_csv}")

    return df_drugs, df_consumption


def seed_database_from_dataset(db_session, df_drugs, df_consumption):
    """
    Populates database tables from the synthetic dataset.
    """
    from backend import models

    # Clear existing data to avoid duplicates if re-seeding
    db_session.query(models.Forecast).delete()
    db_session.query(models.Purchase).delete()
    db_session.query(models.Consumption).delete()
    db_session.query(models.Inventory).delete()
    db_session.query(models.Drug).delete()
    db_session.query(models.Supplier).delete()
    db_session.commit()

    # 1. Insert Suppliers
    supplier_map = {}
    for sup in SUPPLIERS:
        s_obj = models.Supplier(
            supplier_name=sup["supplier_name"],
            contact=sup["contact"],
            lead_time_days=sup["lead_time_days"]
        )
        db_session.add(s_obj)
        db_session.flush()
        supplier_map[sup["supplier_name"]] = s_obj.supplier_id

    # 2. Insert Drugs & Inventory
    drug_id_db_map = {}
    for _, row in df_drugs.iterrows():
        drug_obj = models.Drug(
            drug_name=row["drug_name"],
            category=row["category"],
            manufacturer=row["manufacturer"],
            batch_number=row["batch_number"],
            expiry_date=date.fromisoformat(row["expiry_date"]),
            unit_price=float(row["unit_price"])
        )
        db_session.add(drug_obj)
        db_session.flush()

        drug_id_db_map[row["drug_id"]] = drug_obj.drug_id

        inv_obj = models.Inventory(
            drug_id=drug_obj.drug_id,
            current_stock=int(row["current_stock"]),
            minimum_stock=int(row["minimum_stock"]),
            maximum_stock=int(row["maximum_stock"]),
            reorder_point=int(row["reorder_point"])
        )
        db_session.add(inv_obj)

    # 3. Insert Consumption in bulk batches
    consumption_batch = []
    for _, row in df_consumption.iterrows():
        target_drug_id = drug_id_db_map.get(row["drug_id"])
        if target_drug_id:
            c_obj = models.Consumption(
                drug_id=target_drug_id,
                date=date.fromisoformat(row["date"]),
                quantity_consumed=int(row["quantity_consumed"])
            )
            consumption_batch.append(c_obj)

    db_session.bulk_save_objects(consumption_batch)
    db_session.commit()

    # 4. Insert Initial Purchases (some Pending, some Delivered)
    today = date.today()
    purchases_to_add = []
    for d_id, row in df_drugs.iterrows():
        if row["current_stock"] <= row["reorder_point"]:
            s_name = row["supplier_name"]
            sup_id = supplier_map.get(s_name, list(supplier_map.values())[0])
            lead = row["lead_time_days"]
            real_drug_id = drug_id_db_map[row["drug_id"]]
            status = "Pending" if random.random() < 0.6 else "Delivered"

            p_obj = models.Purchase(
                drug_id=real_drug_id,
                supplier_id=sup_id,
                quantity=int(row["maximum_stock"] - row["current_stock"]),
                purchase_date=today - timedelta(days=random.randint(1, 4)),
                expected_delivery_date=today + timedelta(days=lead),
                status=status
            )
            purchases_to_add.append(p_obj)

    db_session.add_all(purchases_to_add)
    db_session.commit()
    print("Database successfully seeded with suppliers, drugs, inventories, consumptions, and purchase orders.")


if __name__ == "__main__":
    df_d, df_c = generate_full_dataset()
    from backend.database import SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        seed_database_from_dataset(db, df_d, df_c)
    finally:
        db.close()
