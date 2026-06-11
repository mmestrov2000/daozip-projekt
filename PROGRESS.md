# Dnevnik implementacije

Format unosa (max 2 retka po sesiji, kronološki):
- YYYY-MM-DD | taskovi: ... | napravljeno: ... | odstupanja: ... (ili "nema")
- 2026-06-11 | taskovi: F0.3, F0.12 | napravljeno: git mv notebooks/08_drawdown_tree.ipynb i src/supervised.py u archive/ + archive/README.md (negativan rezultat); grep "supervised" čist; pytest zelen | odstupanja: F0.12 bez izmjene — eps već 1.5, git diff prazan
- 2026-06-11 | taskovi: F0.1, F0.2 | napravljeno: dodan config.yaml + src/config.py; konstante u src/utils.py vezane na YAML (PROJECT_START sad 2000-01); pytest pinan u requirements.txt; tests/test_config.py (5 testova, zeleno) | odstupanja: dodane konstante EPSILON_GRID/TC_BPS/MCS_ALPHA/MCS_LOSS uz retke 22–41; pokretanje preko .venv/bin/python (nema python aliasa)
- 2026-06-11 | taskovi: F0.4, F0.5, F0.6 | napravljeno: nema WRDS → grana (b) fja05680/sp500 (stub PROJECT_SPEC.md s „Izvor članstva”); fetch_sp500_membership/membership_on/universe_for_window + sp500_membership.csv (1255 intervala); Stooq fallback + price_source log + 00_price_source_summary.csv (yahoo 759/1080, none 321); testovi membership+prices, pytest 17/17 | odstupanja: ticker_overrides.csv (LEHMQ→LEH, MTLQQ→GM) uveden već u F0.5; Stooq iza anti-bot zaštite → udio stooq 0, fallback validiran mockom
