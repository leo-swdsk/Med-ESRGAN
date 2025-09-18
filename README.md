# Med-ESRGAN: Superresolution für medizinische CT-Bilddaten

Dieses Repository enthält die Arbeitsumgebung für die Bachelorarbeit von Leonard Sawadski.
Der Fokus liegt auf der Anwendung und Anpassung von Superresolution-Modellen für CT-Bilddaten
sowie der Evaluation mit **MSE**, **RMSE**, **MAE**, **PSNR**, **SSIM**, **LPIPS** und **PI**.

---

## 📁 Projektstruktur (empfohlen)

Med-ESRGAN/
├── .venv/                  # Virtuelle Umgebung (nicht im Git enthalten)
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

1) **Repository anlegen/klonen**  
   Erstelle ein leeres Repo „Med-ESRGAN“ (GitHub/GitLab) und klone es lokal.  
   Kopiere deine Skripte/Module sowie `requirements.txt`, `setup.bat`, `setup.sh` hier hinein.

2) **Setup ausführen**

**Windows**
```powershell
.\setup.bat
