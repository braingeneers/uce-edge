PY := .venv/bin/python

core-baseline:
	$(PY) scripts/core_baseline.py

core-onnx-export:
	$(PY) scripts/core_onnx_export.py

core-onnx-quantize:
	$(PY) scripts/core_onnx_quantize.py

core-compare:
	$(PY) scripts/compare.py

core-all: core-baseline core-onnx-export core-onnx-quantize core-compare

core-baseline-33:
	$(PY) scripts/core_baseline.py --ckpt model_files/33l_8ep_1024t_1280.torch --nlayers 33

core-onnx-export-33:
	$(PY) scripts/core_onnx_export.py --ckpt model_files/33l_8ep_1024t_1280.torch --nlayers 33

core-onnx-quantize-33:
	$(PY) scripts/core_onnx_quantize.py --nlayers 33

core-compare-33:
	$(PY) scripts/compare.py --nlayers 33

core-all-33: core-baseline-33 core-onnx-export-33 core-onnx-quantize-33 core-compare-33

brain-baseline:
	$(PY) scripts/brain_baseline.py

brain-onnx-export:
	$(PY) scripts/brain_onnx_export.py

brain-compare:
	$(PY) scripts/brain_compare.py

brain-web-bench:
	$(PY) scripts/brain_web_bench.py

brain-all: brain-baseline brain-onnx-export brain-compare brain-web-bench

brain-extract-embeddings:
	$(PY) scripts/extract_human_protein_embeddings.py

brain-reference-pipeline:
	$(PY) scripts/brain_reference_pipeline.py --n-cells 4

brain-phase01: brain-extract-embeddings brain-reference-pipeline

web-install:
	cd web && npm install

web-typecheck:
	cd web && npx tsc --noEmit

web-build:
	cd web && npx esbuild src/phase2.ts src/phase3.ts src/phase4.ts src/phase5.ts src/phase6.ts src/bench.ts --bundle --format=esm --outdir=dist --sourcemap
	mkdir -p web/dist/ort
	cp web/node_modules/onnxruntime-web/dist/*.wasm web/node_modules/onnxruntime-web/dist/*.mjs web/dist/ort/ 2>/dev/null || true

web-build-phase2: web-build
web-build-phase3: web-build
web-build-phase4: web-build
web-build-phase5: web-build
web-build-phase6: web-build

web-serve:
	$(PY) -m http.server 8765

brain-phase2: web-build
	$(PY) scripts/brain_web_phase2.py

brain-phase3: web-build
	$(PY) scripts/brain_web_phase3.py

brain-phase4: web-build
	$(PY) scripts/brain_web_phase4.py

brain-phase5: web-build
	$(PY) scripts/brain_web_phase5.py

brain-phase6: web-build
	$(PY) scripts/brain_web_phase6.py

brain-bench2: web-build
	$(PY) scripts/brain_web_bench2.py

allen-4layer:
	python UCE/eval_single_anndata.py \
		--adata_path data/allen-celltypes+human-cortex+m1-100.h5ad \
		--dir data/output/ \
		--model_loc model_files/4layer_model.torch \
		--species human \
		--batch_size 8
allen-33layer:
	python UCE/eval_single_anndata.py \
		--adata_path data/allen-celltypes+human-cortex+m1-100.h5ad \
		--dir data/output/ \
		--model_loc model_files/33l_8ep_1024t_1280.torch \
		--species human \
		--nlayers 33 \
		--batch_size 8