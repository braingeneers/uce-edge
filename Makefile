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