# Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport

This repository accompanies the thesis *“Transformers for Tabular Data: A Training Perspective of Self-Attention via Optimal Transport”* (University of Milano-Bicocca, A.Y. 2024/2025).  

The work investigates the training dynamics of **self-attention** through the lens of **Optimal Transport (OT)** and introduces an OT-based alternative model. The thesis combines theoretical insights, simulation studies, and an application to real data.

---

## 📂 Repository Contents
- **Tesi.pdf** – Full thesis document.  
- **functions Transformer.R** – Functions to implement and train Transformer models, including prediction and visualization.  
- **functions OT model.R** – Functions for the Optimal Transport–based model, with prediction and plotting utilities.  
- **simulation.R** – Experimental setups for reproducing the simulation studies described in the thesis.  

---

## ⚙️ Requirements
- **R ≥ 4.2**  
- Suggested packages:  
  - `torch`  
  - `ggplot2`  
  - `dplyr`  
  - `transport` (for Optimal Transport computations)  

Install missing packages in R with:
```r
install.packages(c("ggplot2", "dplyr"))
# torch and transport may require specific installation steps
