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