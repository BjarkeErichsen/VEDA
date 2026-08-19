# How to build GEOM-DRUGS

This directory follows the GEOM-Drugs preparation layout from
[`e3_diffusion_for_molecules`](https://github.com/ehoogeboom/e3_diffusion_for_molecules/tree/main/data/geom).
The raw and processed molecular data are not included because the raw archive
is approximately 50 GB.

1. Download the raw archive from
   [Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?fileId=4360331&version=2.0):

   ```bash
   wget https://dataverse.harvard.edu/api/access/datafile/4360331
   ```

2. Extract the archive into this directory.

3. From the `VEDA-E/` directory, run the local preprocessing script:

   ```bash
   python build_geom_dataset.py \
     --data_dir data/geom \
     --data_file drugs_crude.msgpack \
     --conformations 30 \
     --seed 42
   ```

The tracked `geom_permutation.npy` is the standard GEOM-Drugs split
permutation from the upstream implementation. It contains 6,922,516 molecule
indices and must remain in the same directory as the compatible
`geom_drugs_30.npy` file. The preprocessing script preserves this file when it
is already present.
