import scanpy as sc
import torch
from pathlib import Path
import sys
sys.path.append('/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial')
from stRoamer.utils.data_build import initial_pyG_graph
from stRoamer.inference.run_inference import load_checkpoint, run_inference

base_model_dir='/fs/ess/PAS1475/yzhong/sf_project/base_model/scFM/scGPT_human'
model_path='/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_317M_8_2026-05-23_10-24-41_balanced_quartz/step_150000.pt'
save_dir='/fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human'
adata_dir='/fs/ess/PAS1475/yzhong/sf_project/datasets/spatial_corpus_repo/scGPT_human'

save_path = Path(save_dir)
save_path.mkdir(parents=True, exist_ok=True)
adata_files = sorted(p for p in Path(adata_dir).rglob("*.h5ad") if p.is_file())

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
model = load_checkpoint(path=model_path, 
                        device=device)
model['params']['walk_length']=8    # Length of random walk for model inference

for adata_file in adata_files:
    platform_name = adata_file.parents[0].name
    tissue_name = adata_file.parents[0].parent.name

    adata = sc.read_h5ad(adata_file)
    adata.obs['platform'] = platform_name
    adata.obs['tissue'] = tissue_name

    adata=initial_pyG_graph(
        adata,
        model_dir=base_model_dir,
        gene_col='index',
        batch_size=64,
        preprocess=True
    )

    adata = run_inference(
        input_data=adata,
        model=model,
        device=device,
        batch_size=16,                  # Batch size for model inference
        use_amp=True                    # Whether to use automatic mixed precision
    )
    adata.write_h5ad(save_path / adata_file.name)