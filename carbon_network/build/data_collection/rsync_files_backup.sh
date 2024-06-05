#!/bin/bash

### backup all environmental metatlas files
rsync -avm --include='*.h5' --include='*.parquet' --include='*/' --exclude='*' /global/cfs/cdirs/metatlas/projects/carbon_network/raw_data/global/cfs/cdirs/metatlas/projects/metatlas /global/cfs/cdirs/metatlas/projects/carbon_network/raw_data

### backup all massive files
rsync -avm --include='*.h5' --include='*.parquet' --include='*/' --exclude='*' /pscratch/sd/b/bpb/massive/ /global/cfs/cdirs/metatlas/projects/carbon_network/raw_data/massive

### backup all "no buddy" metatlas files
# rsync -avm --dry-run --include='*.parquet' --include='*/' --exclude='*' /pscratch/sd/b/bpb/metatlas_mdm_parquet_files/global/cfs/cdirs/metatlas/raw_data /global/cfs/cdirs/metatlas/projects/carbon_network/raw_data/metatlas_no_buddy/