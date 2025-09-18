#!/usr/bin/env bash
# setup.sh — Linux/macOS Setup für Med-ESRGAN

# -----------------------------
# Hilfsfunktionen
# -----------------------------
die() { echo "[FEHLER] $*" >&2; exit 1; }

echo "⚙️  Creating virtual environment..."

# -----------------------------
# Python-Interpreter finden
# -----------------------------
if command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  die "Python wurde nicht gefunden. Bitte Python 3.12 installieren und dem PATH hinzufügen."
fi

# -----------------------------
# Python-Version prüfen (muss 3.12.x sein)
# -----------------------------
PYTHON_VERSION="$($PYTHON_CMD --version 2>&1 | awk '{print $2}')"
PY_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
PY_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"

if [[ "${PY_MAJOR}.${PY_MINOR}" != "3.12" ]]; then
  echo "Python 3.12 ist erforderlich, aber gefunden wurde: ${PYTHON_VERSION}"
  exit 1
fi

# -----------------------------
# Virtuelle Umgebung erstellen
# -----------------------------
"$PYTHON_CMD" -m venv .venv || die "Konnte virtuelle Umgebung nicht erstellen."

echo "⚙️  Activating environment and upgrading pip..."
# venv aktivieren
# shellcheck disable=SC1091
source ".venv/bin/activate" || die "Konnte .venv nicht aktivieren."

python -m pip install --upgrade pip || die "Konnte pip nicht aktualisieren."

# -----------------------------
# requirements.txt installieren (falls vorhanden)
# -----------------------------
if [[ -f "requirements.txt" ]]; then
  echo "📦 Installing dependencies from requirements.txt..."
  pip install -r requirements.txt || die "Installation aus requirements.txt fehlgeschlagen."
else
  echo "⚠️  No requirements.txt found. Skipping dependency install."
fi

# -----------------------------
# Git-Submodule initialisieren (wie in setup.bat)
# -----------------------------
echo "⚙️  Initializing git submodules..."
if command -v git >/dev/null 2>&1 && [[ -d ".git" ]]; then
  git submodule update --init --recursive || true
else
  echo "   (Git nicht gefunden oder kein Git-Repo; überspringe Submodules.)"
fi

# -----------------------------
# Ordner anlegen
# -----------------------------
echo "📁  Creating folders..."
mkdir -p data

echo
echo "✅ Setup complete."
echo "To activate your environment, run:"
echo "    source .venv/bin/activate"
