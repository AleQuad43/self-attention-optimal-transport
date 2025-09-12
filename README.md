# Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport

This repository accompanies the thesis *“Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport”* (University of Milano-Bicocca, A.Y. 2024/2025).  

The work investigates the training dynamics of **self-attention** through the lens of **Optimal Transport (OT)** and introduces an OT-based model. The project combines theoretical analysis, simulation studies, and visual diagnostics.

---

## 📂 Repository Contents
- **Tesi.pdf** – Full thesis document.  
- **functions Transformer.R** – Core Transformer functions (building, training, projecting).  
- **functions OT model.R** – Functions for OT-based model training, matching, and prediction.  
- **simulation.R** – End-to-end simulation study setup.  

---

## ⚙️ Requirements
- **R ≥ 4.2**
- Required libraries:  
  ```r
  library(tensorflow)
  library(keras3)
  library(readr)
  library(dplyr)
  library(mvtnorm)
  library(transport)
  library(expm)
