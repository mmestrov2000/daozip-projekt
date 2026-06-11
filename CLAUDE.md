# daozip — upute za rad

Kontrolirana studija stilskog rizika hijerarhijskih alokatora (HRP/HERC/NCO):
korelacijski naspram faktorskog prostora hijerarhije, na point-in-time S&P 500,
s overlay ograničenjem faktorskih beta.

Izvori istine: `TASKS.md` = jedini izvor zadataka i kriterija prihvaćanja;
`PROJECT_SPEC.md` = dizajn i zaključane metodološke odluke; `config.yaml` =
jedino mjesto za parametre (nikad hardkodirati).

Radna pravila:
- Implementiraj isključivo task ID-eve zadane u promptu sesije; ništa izvan toga.
- Prije proglašenja taska gotovim: `pytest` zelen i kriterij prihvaćanja iz TASKS.md ispunjen.
- Po završetku označi checkbox u TASKS.md; svako odstupanje zabilježi u jednom retku ispod taska.
- Ne mijenjaj zaključane odluke, zamrznute panele/izlaze ni dizajn; kod nejasnoće stani i pitaj umjesto da preoblikuješ.
- Najjednostavnije rješenje koje radi: bez refaktoriranja, novih featurea ili apstrakcija izvan opsega taska.
- Na kraju sesije dopiši unos u `PROGRESS.md`.
