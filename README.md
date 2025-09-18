# Med-ESRGAN: Superresolution für medizinische CT-Bilddaten

Dieses Repository enthält die Arbeitsumgebung für die Bachelorarbeit von Leonard Sawadski.
Der Fokus liegt auf der Anwendung und Anpassung von Superresolution-Modellen für CT-Bilddaten
sowie der Evaluation dieser Modelle auf den Metriken **MSE**, **RMSE**, **MAE**, **PSNR**, **SSIM**, **LPIPS** und **PI**.

---

## 📁 Projektstruktur 

Med-ESRGAN/
├── .venv/                  # Virtuelle Umgebung (nicht im Git enthalten)
├── data/                   # Ordner, der für die Original-Daten gedacht ist
├── requirements.txt        # Python-Abhängigkeiten
├── setup.sh                # Setup (Linux/macOS)
├── setup.bat               # Setup (Windows)
├── README.md 
├── Skripte



---

## ⚠️ Voraussetzungen

- Python **3.12** ist installiert und als `python` im Pfad verfügbar.
- (Empfohlen) NVIDIA-GPU mit kompatibler CUDA-Runtime für beschleunigtes Training/Inference.
- Git ist installiert (`git --version`).

---

## 🚀 Erste Einrichtung

1.  **Projekt klonen:**

```bash
git  clone  https://github.com/leo-swdsk/Med-ESRGAN.git

cd  Med-ESRGAN 
```


2.  **Setup-Skript ausführen**

🔧 Linux/macOS
```bash
bash setup.sh
```

🪟 Windows: auf die Setup-Datei klicken oder alternativ
```bash
.\setup.bat
```
---

Wichtig: torch und torchvision werden nicht automatisch installiert
Es muss nach der Aktivierung der virtuellen Umgebung manuell heruntergeladen werden, sofern die Umgebung das unterstützt. 

Manuelle Installation von PyTorch 
```bash
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
```

---
💻 Projektstart (bei jeder weiteren Arbeit)

🔧 Linux/macOS
```bash
source .venv/bin/activate
```

🪟 Windows
```bash
.venv\Scripts\activate
```
---
📥 Datengrundlage & Download
----------------------------

Als Datenbasis dient das öffentlich verfügbare „Spine-Mets-CT-SEG“-Kollektiv („Spine meta-static bone cancer: pre and post radiotherapy CT“) mit 55 CT-Untersuchungen von Patienten mit unterschiedlichen Krebsarten.

**Schritt 1:**  
Besuchen Sie die Webseite unter dem Link  
[https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)  
und drücken Sie unter dem Abschnitt **„Data Access“** auf **„Download (20.36GB)“**.  
So wird die zum Datensatz zugehörige **TCIA-Datei** geladen.

**Schritt 2:**  
Installieren Sie den **NBIA Data Retriever**:

* Entweder unter dem Abschnitt **„Data Access“** auf **„Download requires NBIA Data Retriever“** klicken und den Schritten abhängig vom Betriebssystem folgen,
* **oder** direkt über den Link: [https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) und dann den Schritten abhängig vom Betriebssystem folgen.

**Schritt 3:**  
**NBIA Data Retriever** öffnen und **License Agreement** zustimmen und installieren.

**Schritt 4:**  
Die in **Schritt 1** heruntergeladene **TCIA-Datei** in **NBIA Data Retriever** öffnen und die Dateien in den **/data**\-Ordner herunterladen, der durch das Setup-Skript entstanden ist.

---
## Autor
Bachelorarbeit Medizinische Informatik
**Leonard Sawadski**
Hochschule Heilbronn - 2025