#!/usr/bin/env bash
# scripts/run_all.sh — jedan skriptirani put od sirovih podataka do svih
# tablica i figura koje izvještaj citira (F5.4).
#
# Redoslijed:  pytest  →  nbconvert --execute --inplace  (notebookovi glavnog tijeka).
#
# Uporaba (iz bilo kojeg direktorija — skripta sama prelazi u korijen repoa):
#     bash scripts/run_all.sh
#
# Preduvjet: aktiviran .venv (vidi README §Reprodukcija) i postojeća parquet
# predmemorija cijena u data/raw/price_cache/. Bez predmemorije notebook 01
# prvo preuzima ~757 tickera (Yahoo Finance) i FF5 faktore (Ken French).
#
# Očekivano trajanje (približno; ovisi o hardveru i mreži):
#   • prvo pokretanje (prazna predmemorija, s preuzimanjem):  ~30–40 min
#       — dominira notebook 01 (preuzimanje cijena + FF5 + petfaktorske
#         regresije po prozoru); predmemorija se sprema po dionici u parquet.
#   • ponovno pokretanje (topla predmemorija):                ~12–18 min
#       — sljedeći najskuplji je notebook 02 (bootstrap stabilnost klastera,
#         500 uzoraka, ~5 min); ostali su desetci sekundi do par minuta.
#
# Glavni tijek su točno ovi notebookovi (redom ovisnosti). Izvan tijeka:
#   • 04 (stara evaluacija) i 06 (stari sažetak) — zamijenili su ih 10/12
#     (dijagnoza/mehanizam) i 13 (sažetak); ne proizvode tablice koje
#     izvještaj citira, pa se ne izvršavaju.
#   • 08 (nadzirano proširenje) — arhivirano u archive/.

set -euo pipefail

# --- u korijen repoa, neovisno o tome odakle je skripta pozvana ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# --- odabir Python interpretera: aktivirani venv → projektni .venv → python3 ---
if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
  PY="python"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi
echo ">> Korijen repoa : $ROOT_DIR"
echo ">> Interpreter   : $PY ($("$PY" --version 2>&1))"

# --- notebookovi glavnog tijeka, točnim redom ovisnosti ---
NOTEBOOKS=(
  notebooks/01_data_and_factors.ipynb
  notebooks/02_clustering.ipynb
  notebooks/03_portfolios.ipynb
  notebooks/05_factor_neutral.ipynb
  notebooks/07_clustering_methods.ipynb
  notebooks/09_hierarchical_replication.ipynb
  notebooks/10_hierarchical_diagnosis.ipynb
  notebooks/11_factor_space_intervention.ipynb
  notebooks/12_overlay_mechanism.ipynb
  notebooks/13_final_summary.ipynb
)

SECONDS=0

echo ""
echo "===================================================================="
echo " 1/2  Testovi (pytest)"
echo "===================================================================="
"$PY" -m pytest -q

echo ""
echo "===================================================================="
echo " 2/2  Izvršavanje notebookova (nbconvert --execute --inplace)"
echo "===================================================================="
for nb in "${NOTEBOOKS[@]}"; do
  echo ""
  echo "---- $nb ----"
  nb_start=$SECONDS
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$nb"
  echo "   gotovo za $((SECONDS - nb_start)) s"
done

echo ""
echo "===================================================================="
total=$SECONDS
printf " Sve gotovo. Ukupno trajanje: %dm %02ds\n" $((total / 60)) $((total % 60))
echo " Regenerirane su sve tablice (outputs/tables/) i figure"
echo " (outputs/figures/) koje izvještaj citira."
echo "===================================================================="
