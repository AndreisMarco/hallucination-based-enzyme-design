This repository contains the code for the Master thesis focused expanding the Mosaic, an hallucination-based binder design codebase, with active-site scaffolding capabilities.
The project was supervised by Valentas Brasas and Timothy P. Jenkins at the Digital Biotechnology Lab (DBL) at the Section for Biologics Engineering of DTU.

For an explanation of the Mosaic codebase, refer to the README_mosaic.md (a copy of the README from the original repository).

The main contribution of the thesis can be found in:
* `src/mosaic/losses/`: the files `indexed_scaffolding.py` and `unindexed_scaffolding.py` contain the implementation of the Scaffold object and scaffolding loss terms.
* `src/mosaic/models`: multiple fixes/changes to the protenix and af2 implementations to allow optimization without a target.
* `mosaic_runner/`: contains the scripts necessary to run the scaffolding experiments (also supports binder design, binder design + scaffolding, binder design against multiple targets).
* `src/mosaic/logger.py`: implementation of a logger to keep track of the loss evolution during optimization.

