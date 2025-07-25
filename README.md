# HydroRoot

[![CI status](https://github.com/openalea/hydroroot/actions/workflows/conda-package-build.yml/badge.svg)](https://github.com/openalea/hydroroot/actions/workflows/conda-package-build.yml)
[![Documentation Status](https://readthedocs.org/projects/hydroroot/badge/?version=latest)](https://hydroroot.readthedocs.io/en/latest/?badge=latest)
[![image](https://anaconda.org/openalea3/openalea.hydroroot/badges/version.svg)](https://anaconda.org/openalea3/openalea.hydroroot)

## Software

### Authors

> -   Christophe Pradal
> -   Yann Boursiac
> -   Mikael Lucas
> -   Fabrice Bauget
> -   Christophe Godin
> -   Christophe Maurel

### Institutes

CIRAD / INRAE / inria / CNRS

### Status

Python package

### License

CecILL-C

**URL** : <https://hydroroot.rtfd.io>

## About

### Description

OpenAlea.HydroRoot is a hydraulic root architecture modelling and a root
architecture system generator package.

### Content

The OpenAlea.HydroRoot package contains a root architecture simulation
model coupled with a water and solute transport solver. It contains a
pure hydraulic solver that is solved using resistance network analogy.
It also contains a water and solute transport solver that is more
complex and see the root as a continuous medium.

### Example

Heat map representation of the incoming local radial flows on an
arabidopsis root.

![Alt Text](example/data/fig-6E.png)

### Installation

#### Conda Installation

    conda create -n hydroroot -c conda-forge -c openalea3 openalea.hydroroot

#### Requirements

> -   openalea.mtg
> -   openalea.plantgl
> -   RSML
> -   pandas \> 0.17
> -   numpy
> -   scipy

#### Usage

See notebook in example directory.

## Documentation

<https://hydroroot.rtfd.io>

## Citations

If you use Hydroroot for your research, please cite:

1.  Yann Boursiac, Christophe Pradal, Fabrice Bauget, Mikaël Lucas,
    Stathis Delivorias, Christophe Godin, Christophe Maurel, Phenotyping
    and modeling of root hydraulic architecture reveal critical
    determinants of axial water transport, Plant Physiology, Volume 190,
    Issue 2, October 2022, Pages 1289--1306,
    <https://doi.org/10.1093/plphys/kiac281>
2.  Fabrice Bauget, Virginia Protto, Christophe Pradal, Yann Boursiac,
    Christophe Maurel, A root functional--structural model allows
    assessment of the effects of water deficit on water and solute
    transport parameters, Journal of Experimental Botany, Volume 74,
    Issue 5, 13 March 2023, Pages 1594--1608,
    <https://doi.org/10.1093/jxb/erac471>
