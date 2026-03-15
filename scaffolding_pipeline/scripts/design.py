import jax
import jax.numpy as jnp

from pathlib import Path
from datetime import datetime
import yaml
import argparse
import shutil

import mosaic.losses.motif_scaffolding as ms
from helpers import MODELS, build_loss, build_optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run protein design.")
    parser.add_argument("config", type=Path, help="Path to config yaml file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup directory 
    key = jax.random.key(cfg["seed"])
    run_dir = Path(cfg["dir"]) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    
    # Save a copy of config and input structure
    shutil.copy(args.config, run_dir / "config.yaml")
    shutil.copy(cfg["scaffold"]["input_file"], run_dir / Path(cfg["scaffold"]["input_file"]).name)
    
    # Create scaffold
    scaffold_cfg = cfg["scaffold"]
    scaffold = ms.Scaffold(
        path_to_structure=scaffold_cfg["input_file"],
        keep_intervals=scaffold_cfg["keep_intervals"],
        order=scaffold_cfg["order"],
        loops=scaffold_cfg["loops"],
    )
    scaffold_len = len(scaffold)
    print(f"Scaffold sequence:\n{scaffold.sequence}\n")
    print(f"Scaffold length: {scaffold_len}")

    # Load model and build input features
    model = MODELS[cfg["model"]]()
    design_features, design_structure = model.binder_only_features(binder_length=scaffold_len)

    # Build custom loss function
    loss_cfg = cfg["loss"]
    custom_loss = build_loss(loss_cfg["loss_terms"], scaffold)
    loss_fn = model.build_multisample_loss(
        loss=custom_loss,
        features=design_features,
        recycling_steps=loss_cfg["recycling_steps"],
        num_samples=loss_cfg["num_samples"],
        sampling_steps=loss_cfg["sampling_steps"],
        reduction=jnp.mean,
        initial_recycling_state=None,
        features_to_log=loss_cfg["features_to_log"],
    ) 

    # Optimize
    opt_cfg = cfg["optimizer"]
    key, key_init = jax.random.split(key)
    pssm_init = scaffold.pssm(key=key_init)
    key, key_run_1, key_run_2 = jax.random.split(key, num=3)

    # Exploratory run
    opt = build_optimizer(opt_cfg["run_explore"], loss_fn, scaffold_len)
    pssm, _, trajectory1 = opt.run(pssm_init=pssm_init, update_mask=scaffold.mask, key=key_run_1)
    # Sharpening run
    opt = build_optimizer(opt_cfg["run_sharpen"], loss_fn, scaffold_len)
    final_pssm, best_pssm, trajectory2 = opt.run(pssm_init=pssm, update_mask=scaffold.mask, key=key_run_2)


    # Save trajectory
    log_cfg = cfg["logging"]
    trajectory = trajectory1 + trajectory2
    trajectory.save(
        log_path=run_dir,
        save_loss_plot=log_cfg["save_loss_plot"],
        save_pssm_video=log_cfg["save_pssm_video"],
    )

    # Repredict and save
    key, key_pred = jax.random.split(key)
    pred = model.predict(
        PSSM=final_pssm,
        features=design_features,
        writer=design_structure,
        recycling_steps=4,
        sampling_steps=10,
        initial_recycling_state=None,
        key=key_pred,
    )

    if log_cfg["save_pdb"]:
        pred.save_pdb(path=str(run_dir / "prediction.pdb"))

    print(f"Done. Logs saved to {run_dir.resolve()}")