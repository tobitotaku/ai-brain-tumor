# 🚀 Setup Guide - GBM Classification Pipeline

Voor iedereen die dit project vanaf scratch wil draaien.

---

## 📋 Wat je nodig hebt

- **macOS/Linux/Windows** (macOS instructies hieronder, Linux vergelijkbaar)
- **Python 3.10 - 3.13** (NIET 3.14, zie waarom verderop)
- **Git** geïnstalleerd
- **16GB+ RAM** (liefst 32GB voor snelheid)
- **~10GB schijfruimte** (vooral voor data)

---

## 1️⃣ Pull de Laatste Updates

```bash
# Ga naar je project folder
cd ai-brain-tumor  # of waar je de repo hebt staan

# Pull de retake branch
git fetch origin
git checkout retake/musab
git pull origin retake/musab

# Check dat je op de juiste branch zit
git branch
# Je moet zien: * retake/musab
```

---

## 2️⃣ Python Environment Setup

### Optie A: Met Conda (AANBEVOLEN)

```bash
# Maak environment aan (duurt ~5 min)
conda env create -f environment.yml

# Activeer environment
conda activate gbm-retake

# Check Python versie (moet 3.10-3.13 zijn)
python --version
```

### Optie B: Met venv (als je geen conda hebt)

```bash
# Check Python versie EERST
python3 --version
# Als 3.14 → installeer 3.13: https://www.python.org/downloads/

# Maak virtual environment
python3 -m venv .venv

# Activeer (macOS/Linux)
source .venv/bin/activate

# Activeer (Windows)
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**⚠️ WAAROM NIET PYTHON 3.14?**
- SHAP library werkt niet (numba dependency)
- Gebruik 3.10, 3.11, 3.12 of 3.13

---

## 3️⃣ Data Setup

Je hebt de **raw data** nodig. Er zijn 2 scenario's:

### Scenario A: Je hebt `combined_labeled_standardized.csv` (6.6GB)

```bash
# Zet het bestand hier:
mkdir -p data/raw
# Kopieer combined_labeled_standardized.csv naar data/raw/

# Convert naar nieuwe format
python scripts/convert_old_data.py

# Dit maakt:
# - data/raw/gene_expression.csv (18,635 × 18,859)
# - data/raw/metadata.csv (18,635 × 3)
```

### Scenario B: Je hebt al gene_expression.csv + metadata.csv

```bash
# Zet ze in data/raw/
mkdir -p data/raw
# Kopieer gene_expression.csv naar data/raw/
# Kopieer metadata.csv naar data/raw/

# Je bent klaar! Ga naar stap 4
```

**Check dat data goed staat:**
```bash
ls -lh data/raw/
# Je moet zien:
# - gene_expression.csv (~5-7GB)
# - metadata.csv (~1MB)
```

---

## 4️⃣ Preprocessing

Nu maken we de processed data (filtering, quality checks):

```bash
# Run preprocessing script
python scripts/make_processed.py --config config.yaml
```

**Dit duurt ~2-5 minuten** en maakt:
- `data/processed/expression_processed.csv` (18,635 × 18,752)
- `data/processed/metadata_processed.csv` (18,635 × 2)
- `figures/eda/class_distribution.png`
- `figures/eda/pca_variance.png`
- `reports/tables/data_quality_report.csv`

**Check output:**
```bash
ls -lh data/processed/
# Je moet 2 CSV files zien
```

---

## 5️⃣ Training - Kies je route

### Optie A: Smoke Test (5-10 min) - AANBEVOLEN EERST

Snelle test om te checken dat alles werkt:

```bash
python scripts/train_cv.py --config config_smoke_test.yaml
```

**Dit test:**
- 3×3 nested CV (snel)
- Alleen PCA (geen traag filter_l1)
- 2 modellen (LR + RF)
- ~5-10 minuten

**Als dit werkt → ga door naar Optie B!**

### Optie B: Volledige Academic Run (2-3 uur)

```bash
# BELANGRIJK: Voorkom dat laptop slaapt!
# macOS:
caffeinate -i python scripts/train_cv.py --config config.yaml

# Of in background met log:
caffeinate -i nohup python scripts/train_cv.py --config config.yaml > training.log 2>&1 &

# Check progress:
tail -f training.log
```

**Wat er gebeurt:**
- 5 outer × 3 inner folds = 15 fits per model
- 2 feature routes (filter_l1 + PCA)
- 3 modellen (LR, RF, LightGBM)
- Totaal: 6 combinaties × 15 fits = **90 model trainings**

**Output na ~2-3 uur:**
- `reports/tables/nested_cv_results.csv`
- `reports/tables/summary_metrics.csv` + `.tex`
- `figures/modeling/roc_curves.png`, `pr_curves.png`, etc.
- `figures/calibration/calibration_curves.png`
- `models/final_model_*.pkl`

---

## 6️⃣ Post-Training Analysis (optioneel maar cool)

### A. Feature Stability Analysis

```bash
python scripts/stability_analysis.py
```

**Output:**
- `reports/tables/stability_panel.csv` (top genes met selection frequency)
- `figures/modeling/stability_top50_bar.png`
- `figures/modeling/stability_top50_heatmap.png`

**Dit duurt:** ~15-20 minuten (100 bootstrap iterations)

### B. Compact Gene Panel

```bash
python scripts/shap_compact_panel.py
```

**Output:**
- `models/final_model_compact_panel.pkl`
- `figures/shap/feature_importance_compact_panel.png`
- `reports/tables/metrics_ci_compact_panel.csv`

**Dit duurt:** ~5 minuten

### C. Generate Model Card

```bash
python scripts/generate_model_card.py
```

**Output:**
- `metadata/model_card_generated.md` (auto-populated met resultaten)

**Dit duurt:** <1 minuut

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

```bash
# Check dat je in de juiste environment zit
which python
# Moet wijzen naar .venv/bin/python of conda env

# Re-install requirements
pip install -r requirements.txt
```

### "FileNotFoundError: data/raw/gene_expression.csv"

```bash
# Check dat data er staat
ls data/raw/

# Als niet → ga terug naar stap 3
```

### "MemoryError" tijdens training

```bash
# Je hebt te weinig RAM
# Oplossing 1: Gebruik config_smoke_test.yaml (minder data in geheugen)
# Oplossing 2: Reduceer k_best in config.yaml van 200 → 100
# Oplossing 3: Gebruik alleen PCA route (verwijder filter_l1 uit routes)
```

### Training lijkt te hangen

```bash
# Het hangt niet! Filter_l1 is gewoon traag
# Check progress bar in terminal
# Of check log file:
tail -f training.log

# Je moet zien:
# "Fitting 3 folds for each of X candidates"
# Dit duurt 10-15 min per outer fold
```

### macOS: "libomp not found" (LightGBM error)

```bash
# Install OpenMP
brew install libomp

# Restart terminal
# Try again
```

### Python 3.14 geïnstalleerd maar werkt niet

```bash
# Deactiveer huidige environment
conda deactivate  # of: deactivate

# Install Python 3.13 via conda
conda create -n gbm-retake python=3.13
conda activate gbm-retake
pip install -r requirements.txt
```

---

## 📊 Hoe weet je dat het werkt?

### Na preprocessing:
```bash
ls data/processed/
# Moet zien: expression_processed.csv, metadata_processed.csv

ls figures/eda/
# Moet zien: class_distribution.png, pca_variance.png
```

### Na training (smoke test):
```bash
ls models/
# Moet zien: final_model_pca_*.pkl

ls reports/tables/
# Moet zien: nested_cv_results.csv, metrics_ci_*.csv
```

### Na volledige run:
```bash
# Check aantal resultaten
ls reports/tables/*.csv | wc -l
# Moet minimaal 3-4 CSV files zijn

ls figures/modeling/*.png | wc -l
# Moet minimaal 3 PNG files zijn
```

---

## 🎯 Snelle Start (TL;DR)

## 🎯 Snelle Start (TL;DR)

Als je haast hebt:

```bash
# 1. Pull updates
cd ai-brain-tumor
git checkout retake/musab
git pull origin retake/musab

# 2. Environmentbm-retake

# 3. Data (als je combined_labeled_standardized.csv hebt)
# Zet in data/raw/ en run:
python scripts/convert_old_data.py

# 4. Preprocessing
python scripts/make_processed.py --config config.yaml

# 5. Smoke test
python scripts/train_cv.py --config config_smoke_test.yaml

# 6. Full training (als smoke test werkt)
caffeinate -i nohup python scripts/train_cv.py --config config.yaml > training.log 2>&1 &
tail -f training.log
```

**Totale tijd:** ~3 uur (inclusief downloads en training)

---

## 💡 Pro Tips

1. **Gebruik caffeinate** (macOS) tijdens training:
   ```bash
   caffeinate -i python scripts/train_cv.py --config config.yaml
   ```
   Dit voorkomt dat je laptop slaapt.

2. **Monitor progress** met tail:
   ```bash
   tail -f training.log
   ```

3. **Test eerst met smoke test** voor je 3 uur gaat wachten:
   ```bash
   python scripts/train_cv.py --config config_smoke_test.yaml
   ```

4. **Check disk space** voor je begint:
   ```bash
   df -h .
   # Je hebt ~10GB nodig
   ```

5. **Git stash** je changes voor je pullt:
   ```bash
   git stash
   git pull origin retake/musab
   git stash pop
   ```

---

## 📞 Hulp nodig?

**Check deze files:**
- `README.md` - Algemene docs
- `TRAINING_STATUS.md` - Status van training
- `config.yaml` - Alle settings uitgelegd

**Errors?**
- Check `training.log` voor volledige error message
- Google de error + "scikit-learn" of "pandas"
- Vraag in de groep

**Vragen over code?**
- Alle code heeft docstrings
- Check `src/*.py` voor implementatie details

---

## ✅ Checklist

Voor je begint:
- [ ] Python 3.10-3.13 geïnstalleerd
- [ ] Conda of venv werkend
- [ ] Genoeg disk space (~10GB)
- [ ] Genoeg RAM (16GB minimum)
- [ ] Raw data beschikbaar

Na setup:
- [ ] Environment werkt (`python --version`)
- [ ] Data in `data/raw/`
- [ ] Preprocessing succesvol
- [ ] Smoke test werkt

Klaar voor inleveren:
- [ ] Full training compleet
- [ ] Alle figures gegenereerd
- [ ] Model card gegenereerd
- [ ] Resultaten in `reports/`

---

**Laatst geupdate:** 6 november 2025  
**Door:** Musab Sivrikaya (0988932)

**Succes!** 🚀
