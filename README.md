# g1010
SB3 analysis of Gaia DR3 1010268155897156864 presented in Section 6 of [Tanikawa et al. (2026)](https://arxiv.org/abs/2601.21125).

### data/[subaru_250406_g1010_Yb_R12.csv](https://github.com/kemasuda/g1010/blob/main/data/subaru_250406_g1010_Yb_R12.csv)

- reduced Subaru/HDS spectrum

### [spectrum_fitting.ipynb](https://github.com/kemasuda/g1010/blob/main/spectrum_fitting.ipynb)

- SB3 spectrum fitting presented in Section 6.1.
- results are in spectrum_output/
- this notebook relies on develop branch of [jaxspec](https://github.com/kemasuda/jaxspec/tree/develop)

### [isochrone_fitting.ipynb](https://github.com/kemasuda/g1010/blob/main/isochrone_fitting.ipynb)

- 3-star isochrone fitting presented in Section 6.2.
- results are in isochrone_output/
- this notebook relies on [jaxstar](https://github.com/kemasuda/jaxstar) as well as on develop branch of [jaxspec](https://github.com/kemasuda/jaxspec/tree/develop)
