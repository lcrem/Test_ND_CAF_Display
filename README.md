# DUNE ND-LAr CAF Event Display

This repository provides an interactive 2D/3D event display for interactions within the **DUNE Near Detector Liquid Argon (ND-LAr)** Common Analysis Format (CAF) flat trees. It draws straight lines between the start and end positions of particle hypotheses, color-coded by particle type and overlaid with truth-level information for validation.

The heavy lifting (data parsing, truth-matching, and 3D projection rendering) is managed by the external helper module `ndlar_caf_display_helpers.py`.

---

## Prerequisites

To use this event display, you will need the following Python modules:

- `os`
- `uproot`
- `awkward`
- `matplotlib`
- `numpy`

You will also need one or more **FLAT ND LAr CAFs** (root files).

---

## 1. Loading and Initialising Data

In the first cells, initialise your environment and point the script to your `.root` file:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import sys

# Change below to match your path to NDLAr_CAF_event_display
sys.path.append('/path/to/NDLAr_CAF_event_display/')

# Import everything explicitly from your helper file
from ndlar_caf_display_helpers import load_interaction_spills, plot_interactions
```

Then specify your data file path:

```python
filedir = '/path/to/CAF/files/'
filename = "MicroProdN4p1_NDComplex_FHC.caf.full.spineonly.0002460.CAF.flat.root"
filepath = filedir + '/' + filename
```

---

## 2. Choose Your Reconstruction

The event display will plot straight lines between the start and end position of particle hypotheses. The code was written to accept both **DLP** and **Pandora** reconstruction (although at the moment the DLP option is more mature).

```python
reco_type = "dlp"
# reco_type = "pandora"

spills = load_interaction_spills(filepath, reco=reco_type)
```

---

## 3. Visualisation Modes 🎨

The core visualisation function accepts three distinct configuration modes via the `mode` parameter. Adjust it depending on how much data you want to audit at once.

### Function Parameters Reference 🎛️

`plot_interactions` is configured using the following parameters:

- **`spills`**: The in-memory data object containing the loaded spills extracted from the flat CAF.
- **`spill_index`**: The specific spill or beam slice number you want to view (defaults to `0` for the first spill in the file).
- **`mode`**: Controls how many interactions are plotted at once. Accepted values are:
  - `"single"`: Displays exactly one interaction specified by the `ixn` parameter.
  - `"list"`: Displays only the specific interaction indices passed to `ixn_list`.
  - `"all"`: Displays every single reconstructed interaction found within that spill.
- **`ixn`**: The specific interaction index number to display when running in `mode="single"`.
- **`ixn_list`**: A Python list of interaction index numbers (e.g., `[18, 19, 20]`) to plot simultaneously when running in `mode="list"`.
- **`reco`**: Specifies which reconstruction algorithm data to look up. Set to `"dlp"` for Deep Learning Physics branches or `"pandora"` for Pandora tracking branches.
- **`plot_truth`**: Chooses whether or not to plot truth information (only possible when plotting less than 20 interactions, otherwise it will turn itself to False). Default is `True`.
- **`apply_fv_cut`**: Chooses whether or not to apply a fiducial volume cut (25 cm from all sides of active volume). Default is `False`.
- **`save_dir`**: If defined, saves the event displays into an appropriate directory. Default is `None`.

---

## Example Mode A: Single Event

Zooms in on exactly one interaction index to isolate its trajectories from background noise.

```python
plot_interactions(spills, spill_index=0, mode="single", ixn=45, reco=reco_type, plot_truth=True, save_dir='plots')
```

**Output:**
```
Spill: 0 Interaction: 45 
 Reco: [1 μ⁻, 1 π⁺] (ov=0.95) 
 Truth: νμ CC QE [1 μ⁻, 1 p]
Successfully saved display to: plots/spill_0_single_ixn45.png
```

---

## Example Mode B: Isolated Interaction List

Focuses exclusively on an array of specific interaction indices. This is perfect for troubleshooting truth interactions that get broken up into multiple reco interactions.

```python
plot_interactions(
    spills, 
    spill_index=0, 
    mode="list", 
    ixn_list=[11, 13, 14],
    reco=reco_type,
    plot_truth=True,
    save_dir='plots')
```

**Output:**
```
Spill: 0 Interaction: 11 
 Reco: [1 μ⁻, 1 p] (ov=0.99) 
 Truth: νμ CC COH [1 μ⁻, 1 π⁰, 2 p]
Spill: 0 Interaction: 13 
 Reco: [no reco parts] (ov=0.45) 
 Truth: νμ CC DIS [1 μ⁻, 2 π⁰, 1 p]
Spill: 0 Interaction: 14 
 Reco: [1 μ⁻] (ov=0.99) 
 Truth: νμ CC DIS [1 μ⁻, 1 π⁺, 6 n, 1 p]
Successfully saved display to: plots/spill_0_list_ixn11.png
```

---

## Example Mode C: Complete Spill Overview

Renders every single reconstructed interaction found inside the given spill slice. **Note:** For dense spills, this can look visually cluttered.

```python
plot_interactions(spills, spill_index=0, mode="all", reco=reco_type, plot_truth=False, save_dir='plots')
```

**Output:**
```
NB: More than 20 interactions, switching plot_truth flag to FALSE
Successfully saved display to: plots/spill_0_all_ixn0.png
```

---

## Usage Tips

- Use **Mode A (Single)** for detailed debugging of individual interactions
- Use **Mode B (List)** to compare related interactions side-by-side
- Use **Mode C (All)** for a birds-eye view of all reconstructed activity in a spill
- The overlap (ov) parameter indicates the fraction of truth energy matched to reconstruction
- Debug information includes counts of invalid matches and potential rock muons

---

## Related Files

- `ndlar_caf_display_helpers.py` - Core helper module with data parsing and rendering logic
- `ndlar_caf_display.ipynb` - Original Jupyter notebook with interactive examples

---

## Contact
- Linda Cremonesi (l.cremonesi@imperial.ac.uk) - current lead author, maintener
