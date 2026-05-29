import uproot
import awkward as ak
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
import os
import numpy as np

# ====================================================================
# DEFINITIONS & CONSTANTS
# ====================================================================

GENIE_MODE_NAMES = {
    0:  "Undefined", 1:  "QE", 2:  "RES", 3:  "DIS", 4:  "COH",
    5:  "Elastic", 6:  "IMD", 7:  "AMNuGamma", 8:  "MEC/2p2h",
    9:  "Diffractive", 10: "Coherent π",
}

PDG_TO_PARTICLE = {
    12:  "νe", -12: "ν̄e", 14:  "νμ", -14: "ν̄μ", 16:  "ντ", -16: "ν̄τ",
    11:    "e⁻", -11:   "e⁺", 13:    "μ⁻", -13:   "μ⁺", 22:    "γ",
    211:   "π⁺", -211:  "π⁻", 111:   "π⁰", 2212:  "p",  2112:  "n",
    321:   "K⁺", -321:  "K⁻", 310:   "K_S0", 311:   "K0", 3122:  "λ0",
}

ACTIVE_VOLUME_BOUNDARIES = [[-347.1055, 347.1055], [-215.67131, 81.928696], [ 418.1399, 913.3719]]
FIDUCIAL_VOLUME_INSET = 25

# ====================================================================
# Data Loading & Processing
# ====================================================================

def get_interaction_branches(reco="dlp"):
    reco = reco.lower()
    return [
        f"rec.common.ixn.{reco}..length",
        f"rec.common.ixn.{reco}.part.{reco}..length",
        f"rec.common.ixn.{reco}.part.{reco}..totarraysize",
        f"rec.common.ixn.{reco}.vtx.x", f"rec.common.ixn.{reco}.vtx.y", f"rec.common.ixn.{reco}.vtx.z",
        f"rec.common.ixn.{reco}.part.{reco}.start.x", f"rec.common.ixn.{reco}.part.{reco}.start.y", f"rec.common.ixn.{reco}.part.{reco}.start.z",
        f"rec.common.ixn.{reco}.part.{reco}.end.x", f"rec.common.ixn.{reco}.part.{reco}.end.y", f"rec.common.ixn.{reco}.part.{reco}.end.z",
        f"rec.common.ixn.{reco}.part.{reco}.E", f"rec.common.ixn.{reco}.part.{reco}.pdg",f"rec.common.ixn.{reco}.part.{reco}.primary",
        f"rec.nd.lar.{reco}.ntracks", f"rec.nd.lar.{reco}.nshowers", f"rec.nd.lar.{reco}.tracks..length",
        f"rec.nd.lar.{reco}.tracks.start.x", f"rec.nd.lar.{reco}.tracks.start.y", f"rec.nd.lar.{reco}.tracks.start.z",
        f"rec.nd.lar.{reco}.tracks.end.x", f"rec.nd.lar.{reco}.tracks.end.y", f"rec.nd.lar.{reco}.tracks.end.z",
        f"rec.common.ixn.{reco}.truth..length", f"rec.common.ixn.{reco}.truth..totarraysize", f"rec.common.ixn.{reco}.truth",
        f"rec.common.ixn.{reco}.truth..idx", f"rec.common.ixn.{reco}.truthOverlap", f"rec.common.ixn.{reco}.truthOverlap..length",
        f"rec.common.ixn.{reco}.truthOverlap..idx",
        "rec.mc.nu..length", "rec.mc.nu.id", "rec.mc.nu.mode", "rec.mc.nu.pdg", "rec.mc.nu.E", "rec.mc.nu.iscc",         
        "rec.mc.nu.vtx.x", "rec.mc.nu.vtx.y", "rec.mc.nu.vtx.z", "rec.mc.nu.prim..length", "rec.mc.nu.prim.pdg",
        "rec.mc.nu.prim.end_pos.x", "rec.mc.nu.prim.end_pos.y", "rec.mc.nu.prim.end_pos.z",
        "rec.mc.nu.prim.start_pos.x", "rec.mc.nu.prim.start_pos.y", "rec.mc.nu.prim.start_pos.z",
    ]



def load_interaction_spills(filepath, reco="dlp"):
    file = uproot.open(filepath)
    tree = file["cafTree"]
    return tree.arrays(get_interaction_branches(reco=reco), library="ak")



def get_best_truth_match(event, reco="dlp"):
    reco = reco.lower()
    n_ixn = int(event[f"rec.common.ixn.{reco}..length"])
    n_truth_ids = event[f"rec.common.ixn.{reco}.truth..length"]
    truth_ids = event[f"rec.common.ixn.{reco}.truth"]
    overlaps = event[f"rec.common.ixn.{reco}.truthOverlap"]

    best_truth_id, best_overlap = [], []
    OVERLAP_THRESHOLD = 0.
    
    start_idx = 0
    for i in range(n_ixn):
        end_idx = start_idx + n_truth_ids[i]
        truth_id_arr = np.atleast_1d(truth_ids[start_idx:end_idx])
        overlaps_arr = np.atleast_1d(overlaps[start_idx:end_idx])
        start_idx += n_truth_ids[i]
    
        if len(truth_id_arr) == 0:
            best_truth_id.append(-1)
            best_overlap.append(-1.0)
            continue
    
        idx = np.argmax(overlaps_arr)
        if overlaps_arr[idx] > OVERLAP_THRESHOLD:
            best_truth_id.append(int(truth_id_arr[idx]))
            best_overlap.append(float(overlaps_arr[idx]))
        else:
            best_truth_id.append(-1)
            best_overlap.append(overlaps_arr[idx])

    is_best = [True] * n_ixn
    for i in range(n_ixn):
        if best_truth_id[i] < 0:
            is_best[i] = False
            continue
        for j in range(n_ixn):
            if i != j and best_truth_id[j] == best_truth_id[i] and best_overlap[j] > best_overlap[i]:
                best_overlap[i] = best_overlap[j]
                is_best[i] = False
                    
    return best_truth_id, best_overlap, is_best

# ====================================================================
# Helper functions for internal use in this module
# ====================================================================

def _setup_canvas_and_bounds():
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3)
    ax_xz = fig.add_subplot(gs[0, 0])
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_xy = fig.add_subplot(gs[0, 2])
    ax3d = fig.add_subplot(gs[1:, :], projection="3d")
    
    tpc_bounds = [[low - 50, high + 50] for low, high in ACTIVE_VOLUME_BOUNDARIES]
    fv_bounds = [[low + FIDUCIAL_VOLUME_INSET, high - FIDUCIAL_VOLUME_INSET] for low, high in ACTIVE_VOLUME_BOUNDARIES]
    
    ax3d.set_xlim(tpc_bounds[0][1], tpc_bounds[0][0])
    ax3d.set_zlim(tpc_bounds[1][0], tpc_bounds[1][1])
    ax3d.set_ylim(tpc_bounds[2][0], tpc_bounds[2][1])
    ax3d.view_init(azim=+15, elev=17)
    ax3d.set_box_aspect([1.2, 1, 0.7], zoom=1.3)
    
    return fig, ax_xz, ax_yz, ax_xy, ax3d, tpc_bounds, fv_bounds




def _draw_detector_volumes(ax_xz, ax_yz, ax_xy, ax3d, tpc_bounds, fv_bounds):
    x_min, x_max = ACTIVE_VOLUME_BOUNDARIES[0]
    y_min, y_max = ACTIVE_VOLUME_BOUNDARIES[1]
    z_min, z_max = ACTIVE_VOLUME_BOUNDARIES[2]
    
    x_edges = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
    y_edges = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
    z_edges = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_min]
    ax3d.plot(x_edges, z_edges, y_edges, color='gray', linestyle='--', linewidth=1.5, zorder=0)
        
    ax_xz.set_xlim(tpc_bounds[2][0], tpc_bounds[2][1])
    ax_xz.set_ylim(tpc_bounds[0][0], tpc_bounds[0][1])
    ax_xz.fill_between([z_min, z_max], x_min, x_max, color='lightgray', alpha=0.3, edgecolor='gray', zorder=0)

    ax_yz.set_xlim(tpc_bounds[2][0], tpc_bounds[2][1])
    ax_yz.set_ylim(tpc_bounds[1][0], tpc_bounds[1][1])
    ax_yz.fill_between([z_min, z_max], y_min, y_max, color='lightgray', alpha=0.3, edgecolor='gray', zorder=0)

    ax_xy.set_xlim(tpc_bounds[0][0], tpc_bounds[0][1])
    ax_xy.set_ylim(tpc_bounds[1][0], tpc_bounds[1][1])
    ax_xy.fill_between([x_min, x_max], y_min, y_max, color='lightgray', alpha=0.3, edgecolor='gray', zorder=0)

    # Fiducial Volume Outlines
    x_m_fv, x_M_fv = fv_bounds[0]
    y_m_fv, y_M_fv = fv_bounds[1]
    z_m_fv, z_M_fv = fv_bounds[2]
        
    xf_edges = [x_m_fv, x_M_fv, x_M_fv, x_m_fv, x_m_fv, x_m_fv, x_M_fv, x_M_fv, x_m_fv, x_m_fv, x_M_fv, x_M_fv, x_M_fv, x_M_fv, x_m_fv, x_m_fv]
    yf_edges = [y_m_fv, y_m_fv, y_M_fv, y_M_fv, y_m_fv, y_m_fv, y_m_fv, y_M_fv, y_M_fv, y_m_fv, y_m_fv, y_m_fv, y_M_fv, y_M_fv, y_M_fv, y_M_fv]
    zf_edges = [z_m_fv, z_m_fv, z_m_fv, z_m_fv, z_m_fv, z_M_fv, z_M_fv, z_M_fv, z_M_fv, z_M_fv, z_M_fv, z_m_fv, z_m_fv, z_M_fv, z_M_fv, z_m_fv]
        
    ax3d.plot(xf_edges, zf_edges, yf_edges, color='darkcyan', linestyle=':', linewidth=1.2, alpha=1.)
    ax_xz.fill_between([z_m_fv, z_M_fv], x_m_fv, x_M_fv, color='cyan', alpha=0.1, edgecolor='darkcyan', linestyle=':', zorder=1)
    ax_yz.fill_between([z_m_fv, z_M_fv], y_m_fv, y_M_fv, color='cyan', alpha=0.1, edgecolor='darkcyan', linestyle=':', zorder=1)
    ax_xy.fill_between([x_m_fv, x_M_fv], y_m_fv, y_M_fv, color='cyan', alpha=0.1, edgecolor='darkcyan', linestyle=':', zorder=1)




def _plot_reco_particle(ax3d, ax_xz, ax_xy, ax_yz, g, sx, sy, sz, ex, ey, ez, energy, colour, rec_vtx_x, rec_vtx_y, rec_vtx_z):
    has_bad_endpoint = np.isinf(ex[g]) or np.isnan(ex[g])

    if has_bad_endpoint:
        scale_length = max(15.0, min(energy[g] * 50.0, 40.0)) if energy[g] > 0 else 25.0 
        v_dx, v_dy, v_dz = sx[g] - rec_vtx_x, sy[g] - rec_vtx_y, sz[g] - rec_vtx_z
        norm = np.sqrt(v_dx**2 + v_dy**2 + v_dz**2)
        ux, uy, uz = (v_dx/norm, v_dy/norm, v_dz/norm) if norm > 0 else (0, 0, 1)

        # 3D Shower Cone Surface Mesh
        cone_angles = np.linspace(0, 2 * np.pi, 12)
        cone_lengths = np.linspace(0, scale_length, 5)
        R, THETA = np.meshgrid(cone_lengths, cone_angles)
        cx_mesh = sx[g] + R * ux + (0.20 * R) * np.cos(THETA)
        cy_mesh = sy[g] + R * uy + (0.20 * R) * np.sin(THETA)
        cz_mesh = sz[g] + R * uz
        ax3d.plot_surface(cx_mesh, cz_mesh, cy_mesh, color=colour, alpha=0.4, shade=True, zorder=2)

        # 2D Wedge Patches
        spread = 12.0
        ax_xz.add_patch(Wedge((sz[g], sx[g]), scale_length, np.degrees(np.arctan2(ux, uz)) - spread, np.degrees(np.arctan2(ux, uz)) + spread, facecolor=colour, edgecolor=colour, alpha=0.25, linestyle='--', linewidth=1))
        ax_xy.add_patch(Wedge((sx[g], sy[g]), scale_length, np.degrees(np.arctan2(uy, ux)) - spread, np.degrees(np.arctan2(uy, ux)) + spread, facecolor=colour, edgecolor=colour, alpha=0.25, linestyle='--', linewidth=1))
        ax_yz.add_patch(Wedge((sz[g], sy[g]), scale_length, np.degrees(np.arctan2(uy, uz)) - spread, np.degrees(np.arctan2(uy, uz)) + spread, facecolor=colour, edgecolor=colour, alpha=0.25, linestyle='--', linewidth=1))
    else:          
        ax3d.plot([sx[g], ex[g]], [sz[g], ez[g]], [sy[g], ey[g]], color=colour, linewidth=2)
        ax_xz.plot([sz[g], ez[g]], [sx[g], ex[g]], color=colour)
        ax_xy.plot([sx[g], ex[g]], [sy[g], ey[g]], color=colour)
        ax_yz.plot([sz[g], ez[g]], [sy[g], ey[g]], color=colour)

    ax_xy.scatter(rec_vtx_x, rec_vtx_y, marker='o', s=30, color=colour, edgecolor='black')
    ax_xz.scatter(rec_vtx_z, rec_vtx_x, marker='o', s=30, color=colour, edgecolor='black')
    ax3d.scatter(rec_vtx_x, rec_vtx_z, rec_vtx_y, marker='o', s=30, color=colour, edgecolor='black', linewidth=1.0)
    ax_yz.scatter(rec_vtx_z, rec_vtx_y, marker='o', s=30, color=colour, edgecolor='black')




    
def _plot_true_primaries(ev, ax3d, ax_xz, ax_xy, ax_yz, p_start, p_end, colour, true_vtx_x, true_vtx_y, true_vtx_z):
    ax_xz.scatter(true_vtx_z, true_vtx_x, marker='*', s=30, color=colour, edgecolor='black')
    ax_xy.scatter(true_vtx_x, true_vtx_y, marker='*', s=30, color=colour, edgecolor='black')
    ax_yz.scatter(true_vtx_z, true_vtx_y, marker='*', s=30, color=colour, edgecolor='black')
    ax3d.scatter(true_vtx_x, true_vtx_z, true_vtx_y, marker='*', s=30, color=colour, edgecolor='black', linewidth=1.2)

    ts_x = np.atleast_1d(ev["rec.mc.nu.prim.start_pos.x"][p_start:p_end])
    ts_y = np.atleast_1d(ev["rec.mc.nu.prim.start_pos.y"][p_start:p_end])
    ts_z = np.atleast_1d(ev["rec.mc.nu.prim.start_pos.z"][p_start:p_end])
    te_x = np.atleast_1d(ev["rec.mc.nu.prim.end_pos.x"][p_start:p_end])
    te_y = np.atleast_1d(ev["rec.mc.nu.prim.end_pos.y"][p_start:p_end])
    te_z = np.atleast_1d(ev["rec.mc.nu.prim.end_pos.z"][p_start:p_end])
    
    for p in range(len(ts_x)):
        ax3d.plot([ts_x[p], te_x[p]], [ts_z[p], te_z[p]], [ts_y[p], te_y[p]], color=colour, linewidth=1.5, linestyle=':', alpha=0.8)
        ax_xz.plot([ts_z[p], te_z[p]], [ts_x[p], te_x[p]], color=colour, linewidth=1.5, linestyle=':', alpha=0.8)
        ax_xy.plot([ts_x[p], te_x[p]], [ts_y[p], te_y[p]], color=colour, linewidth=1.5, linestyle=':', alpha=0.8)
        ax_yz.plot([ts_z[p], te_z[p]], [ts_y[p], te_y[p]], color=colour, linewidth=1.5, linestyle=':', alpha=0.8)

def _add_legends(ax3d, legend_entries, legend_labels, plot_truth, reco, apply_fv_cut):

    main_legend = ax3d.legend(legend_entries, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3d.add_artist(main_legend)

    if plot_truth:
        key_handles = [
            plt.Line2D([0], [0], color='dimgray', linestyle='-', linewidth=2),
            plt.Line2D([0], [0], color='none', marker='>', markersize=12, 
                       markerfacecolor='dimgray', markeredgecolor='dimgray', alpha=0.40),
            plt.Line2D([0], [0], color='none', marker='o', markersize=6, markerfacecolor='dimgray', markeredgecolor='gray'),
            plt.Line2D([0], [0], color='dimgray', linestyle=':', linewidth=2),
            plt.Line2D([0], [0], color='none', marker='*', markersize=12, markerfacecolor='dimgray', markeredgecolor='gray')
        ]
        key_labels = ["Reco particle trajectory", "Reco particle shower", "Reco vertex", "True primary trajectory",  "True vertex"]
    else:
        key_handles = [
            plt.Line2D([0], [0], color='dimgray', linestyle='-', linewidth=2),
            plt.Line2D([0], [0], color='none', marker='>', markersize=12, 
                       markerfacecolor='dimgray', markeredgecolor='dimgray', alpha=0.40),
            plt.Line2D([0], [0], color='none', marker='o', markersize=6, markerfacecolor='dimgray', markeredgecolor='gray')
        ]
        key_labels = ["Reco particle trajectory", "Reco particle shower", "Reco vertex"]

    general_legend = ax3d.legend(key_handles, key_labels, loc='upper right', fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', framealpha=0.95)
    ax3d.add_artist(general_legend)

    volume_handles = [
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5),
        Patch(facecolor='cyan', edgecolor='darkcyan', linestyle=':', linewidth=1, alpha=0.25)
    ]
    volume_labels = ["Active Volume Boundary", "Fiducial Volume Region"]

    unified_title = f"Detector Geometry\nReco: {reco.upper()}\nFV Cut: {'ON' if apply_fv_cut else 'OFF'}"
    volume_legend = ax3d.legend(volume_handles, volume_labels, loc='upper left', fontsize=9, title=unified_title, title_fontsize=10, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', framealpha=0.95)
    ax3d.add_artist(volume_legend)




# ====================================================================
# Main High-Level Visualiser Function
# ====================================================================
def plot_interactions(spills, spill_index=0, mode="all", ixn=None, ixn_list=None, reco="dlp", plot_truth=True, apply_fv_cut=False, save_dir=None):
    reco = reco.lower()
    ev = spills[spill_index]

    n_ixn = int(ev[f"rec.common.ixn.{reco}..length"])
    part_len = np.array(ev[f"rec.common.ixn.{reco}.part.{reco}..length"])
    offsets = np.concatenate(([0], np.cumsum(part_len[:-1])))

    best_truth_id, best_overlap, is_best = get_best_truth_match(ev, reco=reco)

    if mode == "all":
        selected = list(range(n_ixn))
    elif mode == "single":
        selected = [ixn]
    elif mode == "list":
        selected = ixn_list
    else:
        raise ValueError("mode must be all, single, or list")
    
    sx, sy, sz = ev[f"rec.common.ixn.{reco}.part.{reco}.start.x"], ev[f"rec.common.ixn.{reco}.part.{reco}.start.y"], ev[f"rec.common.ixn.{reco}.part.{reco}.start.z"]
    ex, ey, ez = ev[f"rec.common.ixn.{reco}.part.{reco}.end.x"], ev[f"rec.common.ixn.{reco}.part.{reco}.end.y"], ev[f"rec.common.ixn.{reco}.part.{reco}.end.z"]
    rx, ry, rz = ev[f"rec.common.ixn.{reco}.vtx.x"], ev[f"rec.common.ixn.{reco}.vtx.y"], ev[f"rec.common.ixn.{reco}.vtx.z"]
    energy_arr = ev[f"rec.common.ixn.{reco}.part.{reco}.E"]
    reco_pdgs = ev[f"rec.common.ixn.{reco}.part.{reco}.pdg"]
    reco_prims = ev[f"rec.common.ixn.{reco}.part.{reco}.primary"]

    fig, ax_xz, ax_yz, ax_xy, ax3d, tpc_bounds, fv_bounds = _setup_canvas_and_bounds()
    _draw_detector_volumes(ax_xz, ax_yz, ax_xy, ax3d, tpc_bounds, fv_bounds)
    
    if apply_fv_cut:
        selected = [I for I in selected if (fv_bounds[0][0] <= rx[I] <= fv_bounds[0][1]) and (fv_bounds[1][0] <= ry[I] <= fv_bounds[1][1]) and (fv_bounds[2][0] <= rz[I] <= fv_bounds[2][1])]
        print(f"FV Cut applied. Kept {len(selected)} interactions.")

    if not selected:
        print("⚠️ No interactions passed the Fiducial Volume cut. Selection skipped.")
        plt.close(fig)
        return

    if plot_truth and len(selected) > 20:
        print("NB: More than 20 interactions, switching plot_truth flag to FALSE")
        plot_truth = False

    colours = ["blue", "red", "green", "purple", "orange", "cyan"]
    legend_entries, legend_labels, possible_rock_muons = [], [], []

    prim_len = np.array(ev["rec.mc.nu.prim..length"])
    prim_offsets = np.concatenate(([0], np.cumsum(prim_len[:-1])))
    
    for c, I in enumerate(selected):
        colour = colours[c % len(colours)]
        start_idx = offsets[I]
        end_idx = offsets[I] + part_len[I]

        ixn_reco_pdgs = np.atleast_1d(reco_pdgs[start_idx:end_idx])
        ixn_reco_prims = np.atleast_1d(reco_prims[start_idx:end_idx])

        # Filter: Keep only the PDG codes where primary == 1 (or True)
        primary_reco_pdgs = ixn_reco_pdgs[ixn_reco_prims == 1]

        # Count unique filtered primary particles
        unique_reco_pdgs, reco_counts = np.unique(primary_reco_pdgs, return_counts=True)
        
        # Build the summary string
        reco_summary = ", ".join([f"{count} {PDG_TO_PARTICLE.get(int(p_code), f'PDG {int(p_code)}')}" for p_code, count in zip(unique_reco_pdgs, reco_counts)])
        
        reco_summary_str = f"Reco: [{reco_summary if reco_summary else 'no reco parts'}]"
        
        truth_interaction_index = best_truth_id[I]
        if truth_interaction_index >= 0:
            genie_mode = GENIE_MODE_NAMES.get(int(ev["rec.mc.nu.mode"][truth_interaction_index]), f"Mode {int(ev['rec.mc.nu.mode'][truth_interaction_index])}")
            nu_name = PDG_TO_PARTICLE.get(int(ev["rec.mc.nu.pdg"][truth_interaction_index]), f"PDG {int(ev['rec.mc.nu.pdg'][truth_interaction_index])}")
            ccnc = "CC" if int(ev["rec.mc.nu.iscc"][truth_interaction_index]) == 1 else "NC"

            if ev["rec.mc.nu.vtx.z"][truth_interaction_index] < ACTIVE_VOLUME_BOUNDARIES[2][0]:
                possible_rock_muons.append(I)

            p_start, p_end = prim_offsets[truth_interaction_index], prim_offsets[truth_interaction_index] + prim_len[truth_interaction_index]
            unique_pdgs, counts = np.unique(np.atleast_1d(ev["rec.mc.nu.prim.pdg"][p_start:p_end]), return_counts=True)
            prim_summary = ", ".join([f"{count} {PDG_TO_PARTICLE.get(int(p_code), f'PDG {int(p_code)}')}" for p_code, count in zip(unique_pdgs, counts)])
            
            leg_text = f"{reco_summary_str} (ov={best_overlap[I]:.2f}) \n Truth: {nu_name} {ccnc} {genie_mode} [{prim_summary if prim_summary else 'no primaries'}]"
            if plot_truth:
                print("Spill:", spill_index, "Interaction:", I, "\n", leg_text)
        else:
            leg_text = f"{reco_summary_str} (Unknown Match)"

        legend_entries.append(plt.Line2D([0], [0], color=colour, linewidth=6))
        legend_labels.append(f"ixn {I}: {leg_text}")
    
        # Reconstructed Particles Loop
        for g in range(start_idx, end_idx):
            _plot_reco_particle(ax3d, ax_xz, ax_xy, ax_yz, g, sx, sy, sz, ex, ey, ez, energy_arr, colour, rx[I], ry[I], rz[I])

        # True Primary Lines
        if plot_truth and truth_interaction_index >= 0:
            _plot_true_primaries(ev, ax3d, ax_xz, ax_xy, ax_yz, p_start, p_end, colour, ev["rec.mc.nu.vtx.x"][truth_interaction_index], ev["rec.mc.nu.vtx.y"][truth_interaction_index], ev["rec.mc.nu.vtx.z"][truth_interaction_index])

    # Finalise Layout Geometry
    ax_xz.set_xlabel("Z [cm]"); ax_xz.set_ylabel("X [cm]"); ax_xz.set_title("XZ Projection")
    ax_yz.set_xlabel("Z [cm]"); ax_yz.set_ylabel("Y [cm]"); ax_yz.set_title("YZ Projection")
    ax_xy.set_xlabel("X [cm]"); ax_xy.set_ylabel("Y [cm]"); ax_xy.set_title("XY Projection")
    ax3d.set_xlabel("X [cm]"); ax3d.set_ylabel("Z [cm]"); ax3d.set_zlabel("Y [cm]"); ax3d.set_title(f"3D Event Display ({reco.upper()})")

    _add_legends(ax3d, legend_entries, legend_labels, plot_truth, reco, apply_fv_cut)

    first_ixn = ixn if mode == "single" else (ixn_list[0] if (mode == "list" and ixn_list) else selected[0])
    
    # Generate and assign the global figure title
    title_text = f"Spill: {spill_index}   |   Mode: {mode.upper()}   |   First ixn Index: {first_ixn}"
    fig.suptitle(title_text, fontsize=13, weight='bold', color='#1a365d', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        first_ixn = ixn if mode == "single" else (ixn_list[0] if (mode == "list" and ixn_list) else selected[0])
        full_save_path = os.path.join(save_dir, f"spill_{spill_index}_{mode}_ixn{first_ixn}.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved display to: {full_save_path}")
    
    plt.show()

    if plot_truth:
        print(f"--- Debug info for ({reco.upper()}) ---")
        print("Number of invalid matches: ", np.sum(np.array(best_truth_id) == -1), "; out of: ", n_ixn)
        print("Number of possible rock muons: ", len(possible_rock_muons), "->", possible_rock_muons)