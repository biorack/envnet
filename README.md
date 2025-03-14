# Environmental Molecular Network (ENVnet)
<img src="envnet_logo.png" alt="EnvNet Logo" style="background-color: white;">


## Understanding the Complexity of Dissolved Organic Matter with ENVnet

Dissolved organic matter (DOM) is a key component of Earth's carbon cycle, but its intricate chemical makeup has been a challenge to scientists. In our recent journal article, we introduce ENVnet—a comprehensive, mass spectrometry-based molecular network of metabolites designed to demystify the complexity of DOM across a variety of environments.

![Example DOM spectrum](/data/DOM-WAVE.png)
<b>Figure 1. An example spectrum of dissolved organic matter as seen in negative ionization mode with high resolution mass spectrometry.</b>  Tens of thousands of unique ions are resolvable in single spectra from this extremely diverse mixture.  By adapting mass difference matching deconvolution with molecular networking, and developing REM-BLINK, a spectral similarity algorithm optimized for DOM, we overcame the challenge of chimeric MS/MS (data produced from more than one molecule) which limited molecularly resolved analysis of DOM to create ENVnet.

ENVnet allows us to uncover globally distributed molecules with conserved structures, offering fresh insights into DOM's composition. Notably, it highlights the presence of long-lasting carboxylated cyclic molecules that persist in different ecosystems. With ENVnet, researchers can now test hypotheses concerning specific molecular structures involved in DOM cycling.

This repository not only advances our understanding of DOM but also provides a valuable resource for the scientific community to explore the molecular intricacies of Earth's carbon processes.

ENVnet can be used in your custom Python applications, but we highly recommend using the workflow in GNPS2 (http://www.gnps2.org)

[![Example Network](/data/cosmograph_screenshot.png?raw=true "Optional Title")](https://cosmograph.app/run/?data=https://raw.githubusercontent.com/biorack/envnet/main/data/edge_data.csv&source=source&target=target&gravity=0.25&repulsion=1&repulsionTheta=1.15&linkSpring=0.16&linkDistance=10&friction=0.85&renderLabels=true&renderHoveredLabel=true&renderLinks=true&nodeSizeScale=1&linkWidthScale=1&linkArrowsSizeScale=1&nodeSize=size-total%20links&nodeColor=color-total%20links&linkWidth=width-default&linkColor=color-default&)


Explore interactive ENVnet visualization with [Cosmograph here](https://cosmograph.app/run/?data=https://raw.githubusercontent.com/biorack/envnet/main/data/edge_data.csv&source=source&target=target&gravity=0.25&repulsion=1&repulsionTheta=1.15&linkSpring=0.16&linkDistance=10&friction=0.85&renderLabels=true&renderHoveredLabel=true&renderLinks=true&nodeSizeScale=1&linkWidthScale=1&linkArrowsSizeScale=1&nodeSize=size-total%20links&nodeColor=color-total%20links&linkWidth=width-default&linkColor=color-default&)
