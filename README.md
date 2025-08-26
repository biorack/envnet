# ENVnet

ENVnet is an open-source, repository-scale resource for mass spectrometry-based molecular networking of dissolved organic matter (DOM) across diverse environments. The complexity of DOM has long limited our ability to understand its role in Earth's carbon cycle, especially at the molecular level. ENVnet addresses this challenge by integrating high-quality tandem mass spectrometry data from thousands of samples, applying advanced computational deconvolution and molecular networking, and providing a durable, version-controlled platform for comparative and integrative analysis. By enabling molecularly resolved insights into DOM composition, ENVnet empowers researchers to explore hypotheses about carbon cycling, microbial degradation, and land management with unprecedented resolution.

ENVnet is organized into four main modules, each supporting a distinct capability in the workflow:

---

## 1. Deconvolution

The **Deconvolution** module is designed to resolve chimeric MS/MS spectra, a major obstacle in DOM analysis. Using a curated list of neutral losses and a novel, high-performance algorithm based on mass difference matching (MDM), this module separates overlapping signals to produce pure spectra suitable for downstream annotation and network construction.

**Technical Details:**  
- Implements MDM-based deconvolution optimized for negative ionization mode data.
- Utilizes a defined set of neutral losses to guide spectral separation.
- Includes preprocessing and postprocessing routines for quality control.
- Validates deconvoluted spectra against MS1 data to ensure precursor accuracy.
- Outputs high-quality, non-redundant spectra for library searching and network building.

---

## 2. Build

The **Build** module constructs the ENVnet molecular network resource from the best available collection of curated files. This process integrates new and public datasets, applies clustering and formula assignment, and produces a version-controlled network that can be distributed and updated as new data becomes available.

**Technical Details:**  
- Aggregates and cleans large-scale MS/MS datasets from diverse environments.
- Performs clustering and molecular formula assignment using MS-BUDDY and other tools.
- Matches spectra to reference libraries and annotates nodes with chemical class information.
- Builds and visualizes molecular networks using optimized algorithms (e.g., REM-BLINK).
- Maintains ENVnet as a durable, updatable resource for comparative studies.

---

## 3. Annotation

The **Annotation** module links new experimental data to the ENVnet resource, enabling researchers to contextualize their findings within a global molecular network. This module automates the matching of MS1 and MS2 features, propagates annotations, and supports batch processing for high-throughput studies.

**Technical Details:**  
- Matches new spectra to ENVnet nodes using spectral similarity and formula assignment.
- Propagates chemical class and pathway annotations across the network.
- Integrates external databases and supports batch annotation workflows.
- Facilitates comparative analysis between experiments and the ENVnet reference.

---

## 4. Analysis

The **Analysis** module provides statistical and multivariate tools for interpreting annotated molecular networks and environmental datasets. It supports enrichment analysis, visualization, and hypothesis testing, enabling researchers to uncover patterns and relationships in DOM composition and dynamics.

**Technical Details:**  
- Offers statistical routines for differential analysis and enrichment testing.
- Includes multivariate analysis tools for clustering, dimensionality reduction, and pattern discovery.
- Provides visualization utilities for molecular networks and annotation results.
- Supports reproducible workflows for data exploration and export.

---

## Additional Sections for README

- **Installation**: How to install ENVnet and its dependencies.
- **Quick Start**: Example workflow to get started.
- **Documentation**: Links to API documentation and tutorials.
- **Contributing**: Guidelines for contributing to ENVnet.
- **License**: Licensing information.
- **Contact/Support**: How to get help or report issues.

---

## Installation

```bash
git clone https://github.com/yourusername/envnet.git
cd envnet
pip install -e .
```

## Quick Start

```python
import envnet
from envnet.deconvolution import workflows

# Run a deconvolution workflow on your data
results = workflows.run_deconvolution('path/to/datafile')
```

## Documentation

See [docs/](docs/) for API documentation and tutorials.

## Contributing

Pull requests and issues are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed according to the academic/nonprofit license in this repository.

## Contact

For questions or support, open an issue on GitHub or contact the maintainers.