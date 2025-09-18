## Anleitung: ESRGAN‑Med – End‑to‑End Workflow (für PDF‑Export geeignet)

Hinweis: Führen Sie alle Befehle aus dem Ordner `ESRGAN` aus.

```bash
cd ESRGAN
```

### Überblick
- **Ziel**: CT‑DICOMs vereinheitlichen (Pixel Spacing), trainieren (Vortraining + Feintuning), evaluieren (Testsplit), visualisieren (LR vs SR vs HR).
- **Artefakte**: Resampled Daten in `ESRGAN/preprocessed_data/`, Trainingsläufe in `ESRGAN/runs/`, Evaluationsergebnisse in `ESRGAN/eval_results/`.
- **Setup**: Bitte das Projekt gemäß `README.md` einrichten (Abhängigkeiten, Datenpfade, Umgebungen).

---

## 1) Vorverarbeitung: homogenes Pixel Spacing
Script: `preprocess_resample_ct.py`

Kurzbeschreibung:
- Resampled CT‑DICOMs auf einheitliches in‑plane Pixel Spacing (Standard: 0.8 mm) und legt das Ergebnis standardmäßig unter `ESRGAN/preprocessed_data/` ab.

Minimaler Befehl:
```bash
python preprocess_resample_ct.py --root "..\data\manifest-1724965242274\Spine-Mets-CT-SEG"
```

Wichtige Optionen:
- `--out_dir`: Standard ist `ESRGAN/preprocessed_data`.
- `--target_spacing`: Standard 0.8 (mm).

---

## 2) Patientenaufteilung (Train/Val/Test) erzeugen
Script: `dump_patient_split.py`

Kurzbeschreibung:
- Erstellt deterministische Patienten‑Splits (Standard‑Seed 42) und schreibt die JSON nach `ESRGAN/splits/patient_split_seed42.json`.

Minimaler Befehl:
```bash
python dump_patient_split.py --root "preprocessed_data"
```

Wichtige Optionen:
- `--seed`: Standard 42; `--output`: Standardpfad unter `ESRGAN/splits/`.

---

## 3) Vortraining (Pretraining)
Script: `train_ct_sr.py`

Kurzbeschreibung:
- Trainiert das RRDB‑Modell mit L1‑Loss (ohne GAN). Artefakte (Checkpoints, Logs) landen automatisch unter `ESRGAN/runs/<laufname>/`.

Minimaler Befehl:
```bash
python train_ct_sr.py
```

Wichtige Optionen:
- `--data_root`: Standard `ESRGAN/preprocessed_data`.
- `--scale`: Standard 2; `--epochs`: Standard 50; `--batch_size`: Standard 10; `--patch_size`: Standard 192.
- Degradation (Standard `blurnoise`):
  - `--noise_sigma_range_norm`: Standard `0.001 0.003` (Rauschen auf [-1,1]‑Normierung; ca. bis ~10 HU)
  - `--dose_factor_range`: Standard `0.25 0.5` (Rauschen ~ 1/√dose; kleinere Werte ⇒ mehr Rauschen)
  - `--blur_sigma_range`: Standard automatisch nach `--scale` (x2≈0.8±0.1, x4≈1.2±0.15)
  - `--blur_kernel`: Standard automatisch aus σ (ungerade Größe)
  - `--antialias_clean`: nur relevant für `clean`
- Weitere Degradationsmodi:
  - `clean`: reines Downsampling (optional mit `--antialias_clean`)
  - `blur`: nur Unschärfe (ohne zusätzliches Rauschen)
- Hinweis: 
    - Skalierungsfaktor 4 nicht funktionsfähig, da nur teilweise im Code vorbereitet 

---

## 4) Feintuning (Finetuning)
Script: `finetune_ct_sr.py`

Kurzbeschreibung:
- Feintuning mit Wahrnehmungs‑ und GAN‑Termen auf Basis eines vortrainierten Generators.

Minimaler Befehl (erfordert nur den Pfad zum vortrainierten Generator):
```bash
python finetune_ct_sr.py --pretrained_g "runs\rrdb_x2_blurnoise_20250912-114004\best.pth"
```

Wichtige Optionen:
- `--data_root`: Standard `ESRGAN/preprocessed_data`; `--scale`: Standard 2; `--epochs`: Standard 10.
- `--warmup_g_only`: Standard 100 Iterationen nur G; `--lambda_perc`: 0.08; `--lambda_gan`: 0.003.
- Degradation (Standard `blurnoise`): gleiche Defaults wie im Vortraining (siehe oben).

---

## 5) Evaluation (auf Testsplit)
Script: `evaluate_ct_model.py`

Kurzbeschreibung:
- Berechnet Metriken (z. B. MAE/PSNR) und schreibt CSV/JSON sowie Plots nach `ESRGAN/eval_results/`.

Minimaler Befehl (Testsplit wie in der Arbeit analysiert):
```bash
python evaluate_ct_model.py --root "preprocessed_data" --split test --model_path "runs\finetune_x2_blurnoise_20250914-093436\best.pth"
```

Wichtige Optionen:
- `--preset` bzw. `--window_center/--window_width` für Metrik‑Fensterung (Standard keine Preset‑Erzwingung, globale HU‑Spanne für Degradation).
- Degradation‑Defaults: `blurnoise`; Sampling‑Modus standardmäßig `volume`.
- `--device`: Standard `cuda`. Für reine CPU‑Läufe kann `--device cpu` gesetzt werden.

---

## 6) Visualisierung (LR vs SR vs HR)
Script: `visualize_lr_sr_hr.py`

Kurzbeschreibung:
- Interaktiver Viewer (Mausrad fürs Scrollen). Standard‑Fensterung: `soft_tissue`.

Minimaler Befehl (nur Patientenordner und Modellpfad setzen):
```bash
python visualize_lr_sr_hr.py --dicom_folder "preprocessed_data\15041pp" --model_path "runs\finetune_x2_blurnoise_20250914-093436\best.pth"
```

Wichtige Optionen:
- `--preset`: Standard `soft_tissue`; `--device`: Standard `cuda`; `--scale`: Standard 2.
- Degradation im Viewer: Standard `blurnoise` (einheitlich pro Volume angewandt).

---

## 7) Kurz: Weitere hilfreiche Skripte

### 7.1) LR vs HR (ohne SR) – schneller Vergleich
Script: `visualize_lr_hr.py`

Minimaler Befehl:
```bash
python visualize_lr_hr.py --dicom_folder "preprocessed_data\15041pp"
```
Standard: `preset` soft_tissue, `scale` 2, Degradation `blurnoise`.

### 7.2) Modellparameter zählen und (optional) Aufwand schätzen
Script: `count_model_params.py`

Beispiel (mit geladenem Checkpoint):
```bash
python count_model_params.py --scale 2 --model_path "runs\rrdb_x2_blurnoise_20250912-114004\best.pth"
```
Optional: `--profile` zur MACs/FLOPs‑Schätzung; Standard‑Eingabegröße 256×256 (LR).

---

## Hinweise zur Reproduzierbarkeit
- Standard‑Seeds sind gesetzt (z. B. 42) und konsistent zwischen Split, Training und Evaluation.
- Pfade sind relativ zum Ordner `ESRGAN` gewählt. Bitte die Beispiel‑Pfadwerte ggf. an eigene Laufordner anpassen.
- Für CPU‑Runs kann überall `--device cpu` gesetzt werden; dies ist langsamer und dient nur der Funktionsprüfung.


