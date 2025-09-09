# Autoencoder Derived RSEI Improves Long Term Forest Condition Detection in Khyber Pakhtunkhwa (2001–2023)


## Overview

This repository contains the materials supporting the manuscript:

**"Autoencoder Derived RSEI Improves Long Term Forest Condition Detection in Khyber Pakhtunkhwa (2001–2023)"**

The study develops and evaluates an autoencoder-based Remote Sensing Ecological Index (AERSEI) using a 23-year Landsat archive. AERSEI is benchmarked against the conventional PCA-based RSEI, demonstrating superior ability to capture nonlinear ecological interactions, detect subtle degradation/recovery trends, and attribute spatial drivers using explainable machine learning (SHAP).

---

## Key Contributions

- **Novel Index**: Developed a shallow autoencoder–derived RSEI (AERSEI) integrating greenness, moisture, dryness, thermal, and topographic indicators.  
- **Comparative Benchmarking**: Systematic comparison against PCA-RSEI showed stronger alignment with kNDVI and enhanced sensitivity to ecological changes.  
- **Trend Analysis**: Identified 8.9% degradation and 5.1% recovery across KPK forests (2001–2023), compared to near-null detection with PCA.  
- **Driver Attribution**: Applied XGBoost + SHAP to disentangle biophysical and anthropogenic controls (elevation, precipitation, LST, soil moisture).  
- **Uncertainty Assessment**: Bootstrap ensembles provided confidence intervals, highlighting regions of elevated predictive uncertainty.  

---

## Repository Structure


---

## Data Sources

- **Landsat (2001–2023)**: TM, ETM+, and OLI surface reflectance (USGS Collection 2 Tier 1).  
- **Topography**: SRTM DEM.  
- **Derived Variables**: kNDVI, NDVI, NDBSI, IBI, Wetness, LST, Soil Moisture Index.  
- **Forest Types**: Five-category classification (needleleaf, broadleaf, deciduous, shrubland, grassland).

*Note*: Due to data policy, raw Landsat imagery is not included. Scripts are provided to reproduce preprocessing in **Google Earth Engine (GEE)**.

---

## Methods

1. **Preprocessing**: Annual cloud-free, growing-season composites (April–October) created in GEE.  
2. **Index Derivation**:  
   - Autoencoder (AE) latent feature → AERSEI  
   - PCA first component → PCA-RSEI  
3. **Statistical Evaluation**: Correlation, RMSE, concordance, effect size, variance diagnostics.  
4. **Trend Detection**: Trend-Free Prewhitened Mann–Kendall (TFPW-MK) with FDR correction.  
5. **Attribution**: XGBoost regression with SHAP feature interpretation.  
6. **Uncertainty**: Bootstrap ensemble modeling (B=100).  

---

## Requirements

- **Languages**:  
  - Python ≥ 3.9 (TensorFlow/Keras, XGBoost, SHAP, NumPy, Pandas, Matplotlib)  
  - R ≥ 4.2 (terra, dplyr, Kendall, SHAPforxgboost, ggplot2, gridExtra)

- **Platforms**:  
  - Google Earth Engine (for Landsat preprocessing)  
  - RStudio / Jupyter for subsequent analysis  

---


