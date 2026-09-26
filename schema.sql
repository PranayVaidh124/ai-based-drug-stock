-- =======================================================
-- AI-Based Drug Stock & Supply Chain Optimization System
-- Database Schema (MySQL / MariaDB / phpMyAdmin / Workbench)
-- =======================================================

CREATE DATABASE IF NOT EXISTS drug_stock_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE drug_stock_db;

-- 1. Table: suppliers
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_name VARCHAR(150) NOT NULL,
    contact VARCHAR(100),
    lead_time_days INT NOT NULL DEFAULT 7,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_supplier_name (supplier_name)
) ENGINE=InnoDB;

-- 2. Table: drugs
CREATE TABLE IF NOT EXISTS drugs (
    drug_id INT AUTO_INCREMENT PRIMARY KEY,
    drug_name VARCHAR(150) NOT NULL UNIQUE,
    category VARCHAR(100) NOT NULL,
    manufacturer VARCHAR(150),
    batch_number VARCHAR(50),
    expiry_date DATE NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_drug_category (category),
    INDEX idx_expiry_date (expiry_date)
) ENGINE=InnoDB;

-- 3. Table: inventory
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INT AUTO_INCREMENT PRIMARY KEY,
    drug_id INT NOT NULL UNIQUE,
    current_stock INT NOT NULL DEFAULT 0,
    minimum_stock INT NOT NULL DEFAULT 10,
    maximum_stock INT NOT NULL DEFAULT 500,
    reorder_point INT NOT NULL DEFAULT 20,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_inventory_drug FOREIGN KEY (drug_id) REFERENCES drugs(drug_id) ON DELETE CASCADE,
    INDEX idx_current_stock (current_stock),
    INDEX idx_reorder_point (reorder_point)
) ENGINE=InnoDB;

-- 4. Table: consumption
CREATE TABLE IF NOT EXISTS consumption (
    consumption_id INT AUTO_INCREMENT PRIMARY KEY,
    drug_id INT NOT NULL,
    date DATE NOT NULL,
    quantity_consumed INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_consumption_drug FOREIGN KEY (drug_id) REFERENCES drugs(drug_id) ON DELETE CASCADE,
    INDEX idx_consumption_drug_date (drug_id, date)
) ENGINE=InnoDB;

-- 5. Table: purchases
CREATE TABLE IF NOT EXISTS purchases (
    purchase_id INT AUTO_INCREMENT PRIMARY KEY,
    drug_id INT NOT NULL,
    supplier_id INT NOT NULL,
    quantity INT NOT NULL,
    purchase_date DATE NOT NULL,
    expected_delivery_date DATE,
    status VARCHAR(50) NOT NULL DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_purchases_drug FOREIGN KEY (drug_id) REFERENCES drugs(drug_id) ON DELETE RESTRICT,
    CONSTRAINT fk_purchases_supplier FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE RESTRICT,
    INDEX idx_purchases_status (status),
    INDEX idx_purchases_drug (drug_id),
    INDEX idx_purchases_supplier (supplier_id)
) ENGINE=InnoDB;

-- 6. Table: forecasts
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INT AUTO_INCREMENT PRIMARY KEY,
    drug_id INT NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_demand FLOAT NOT NULL DEFAULT 0.0,
    confidence_level VARCHAR(50) NOT NULL DEFAULT 'Normal',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_forecasts_drug FOREIGN KEY (drug_id) REFERENCES drugs(drug_id) ON DELETE CASCADE,
    INDEX idx_forecasts_drug_date (drug_id, forecast_date)
) ENGINE=InnoDB;
