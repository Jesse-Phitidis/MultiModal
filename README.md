# Hetero-modal learning and corruption-resistant hetero-modal inference for joint segmentation of white matter hyperintensities and ischaemic stroke lesions in MRI

This repository contains the code for the paper:

> _To be added_

## Installation

Create the conda environment from the provided environment definition:

```bash
conda env create -f environment.yaml
conda activate mm
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

## MMAR implementation

The code for the MMAR is in [`src/multimodal/models/transformer_feature_router_updated.py`](src/multimodal/models/transformer_feature_router_updated.py), in the `TransformerFeatureRouter` class.

## Run training

To perform a pytorch lightning training run create a config and run: 
```bash
scripts/main.py fit --config path_to_config.yaml
```
