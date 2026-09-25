# Fixtures

`fixtures/room1_*.png` are five 512×512 16-bit frames from **TUM-VI `dataset-room1_512_16`**
(`mav0/cam0|cam1/data/<ns>.png`; CC BY 4.0, TU Munich — see `tools/datasets/`).
`c0_a`/`c1_a` are the stereo pair at t=1520530358151567223, `c0_b` is cam0 250 ms later,
`c0_c`/`c0_d` are 30 s and 60 s later. The test converts them to 8-bit mono and resizes
to each export resolution. `pytest -q tests` after the four `export.py` runs (see T-0109).
