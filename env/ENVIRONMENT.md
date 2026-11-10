# Environment lock (Phase 0, 2026-11-07)

Machine: i9-13900K (8 P-cores + 16 E-cores, 32 threads), 64 GB RAM, Windows 11 Pro 10.0.26300 (native; no WSL).

## Python (uv venv D:\_caches\envs\knitplan, CPython 3.13.13)

```
alembic==1.20.1
annotated-types==0.8.0
antlr4-python3-runtime==4.13.2
autoray==0.11.0
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
colorlog==6.12.0
contourpy==1.4.0
cotengra==0.8.3
cryptography==50.0.2
cycler==0.12.1
dill==0.4.1
duckdb==2.0.0
flatbuffers==25.12.19
fonttools==4.68.0
ibm-cloud-sdk-core==3.28.2
ibm-platform-services==0.77.2
ibm-quantum-schemas==0.13.20261016
idna==3.20
joblib==1.6.0
kiwisolver==1.5.1
mako==1.4.3
markupsafe==3.0.4
matplotlib==3.11.2
mqt-bench==2.3.0
narwhals==2.28.0
networkx==3.7
numpy==2.5.4
onnxruntime==1.31.1
openqasm3==1.0.1
opt-einsum==3.4.0
optuna==5.0.0
orjson==3.13.0
packaging==26.3
pandas==3.1.1
pillow==13.0.0
protobuf==7.36.2
psutil==7.2.2
pybase64==1.5.1
pycparser==3.12
pydantic==2.14.3
pydantic-core==2.50.4
pyjwt==2.15.1
pymupdf==2.1
pymupdf-layout==2.1
pymupdf4llm==2.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==3.0.0
pyyaml==6.0.3
qiskit==2.6.2
qiskit-addon-cutting==0.10.0
qiskit-aer==0.18.2
qiskit-ibm-runtime==0.51.0
qiskit-mitigation==0.2.0
qiskit-qasm3-import==0.6.0
requests==2.34.2
requests-ntlm==1.3.0
rustworkx==0.18.1
samplomatic==0.21.0
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
sqlalchemy==2.1.8
sspilib==0.6.0
stevedore==5.9.1
tabulate==0.10.0
threadpoolctl==3.8.0
tqdm==4.70.1
typing-extensions==4.16.0
typing-inspection==0.4.5
tzdata==2026.6
urllib3==2.8.0
```

## Julia

- Julia 1.13.3 portable (D:\_caches\julia-bin, sha256 of zip 1fae8fc95e6199b3d8f6d7a3791810e45e1b3c480eb4b9f5bb211789badfc729), depot D:\_caches\julia
- Galley.jl 0.2.2 @ 8eefbb52c64c9dd1f7cad39e30c9541a9ab9dc56 (developed from the local clone)
- Finch.jl 2.1.1, JSON, BenchmarkTools (Manifest: D:\runs\knitplan\galley_env\Manifest.toml)

## External code (pinned clones in D:\runs\knitplan\ext)

- CutQC github.com/weiT1993/CutQC @ cda68c7bc1e7fb9ea403d78b759ed5bdb4d00e34
- QCut github.com/FiQCI/QCut @ 8b2ee8f58f5f8198c07cd583c7e1ffb7a8d03fd2 (v2.4.0-3-g8b2ee8f); D5 installs tag v2.4.0

## Toolchains

