# Dnevnik implementacije

Format unosa (max 2 retka po sesiji, kronološki):
- YYYY-MM-DD | taskovi: ... | napravljeno: ... | odstupanja: ... (ili "nema")
- 2026-06-11 | taskovi: F0.3, F0.12 | napravljeno: git mv notebooks/08_drawdown_tree.ipynb i src/supervised.py u archive/ + archive/README.md (negativan rezultat); grep "supervised" čist; pytest zelen | odstupanja: F0.12 bez izmjene — eps već 1.5, git diff prazan
- 2026-06-11 | taskovi: F0.1, F0.2 | napravljeno: dodan config.yaml + src/config.py; konstante u src/utils.py vezane na YAML (PROJECT_START sad 2000-01); pytest pinan u requirements.txt; tests/test_config.py (5 testova, zeleno) | odstupanja: dodane konstante EPSILON_GRID/TC_BPS/MCS_ALPHA/MCS_LOSS uz retke 22–41; pokretanje preko .venv/bin/python (nema python aliasa)
