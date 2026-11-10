"""Visualize a median-disagreement repetition, selected without classifier scores."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import ROOT, HERE, DATA, load_json, core
from fusion import align_view, fuse_segments

EDGES = [(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(12,14),(14,16),(16,18),(16,20),(16,22),
         (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]


def main():
    observations = load_json(DATA / "observations.json")
    diagnostics = load_json(DATA / "fusion_diagnostics.json")
    order = sorted(range(len(diagnostics)), key=lambda i: diagnostics[i]["aligned_spread_mean"])
    i = order[len(order)//2]
    row = observations[i]
    assert row["sample_id"] == diagnostics[i]["sample_id"]
    segments = [np.load(ROOT / "data/REHAB26-ViewAngles/raw" / view["skeleton_file"], mmap_mode="r")
                [row["first_frame"]:row["last_frame"]+1] for view in row["source_views"]]
    aligned = [align_view(s) for s in segments]
    fused = fuse_segments(segments)
    frames = [r[0][48] for r in aligned] + [fused["xyz"][48]]
    combined = np.stack(frames)[:, 11:]
    center = (combined.max((0,1)) + combined.min((0,1))) / 2
    radius = np.ptp(combined, axis=(0,1)).max() / 2 * 1.1
    fig = plt.figure(figsize=(13,4.3), layout="constrained")
    for n, (xyz, title) in enumerate(zip(frames, ["Aligned camera 1", "Aligned camera 2", "Aligned camera 3", "Visibility-squared fusion"])):
        ax = fig.add_subplot(1,4,n+1,projection="3d")
        for a,b in EDGES:
            ax.plot(xyz[[a,b],0],xyz[[a,b],2],-xyz[[a,b],1],color="#347f80" if n==3 else "#677e99",lw=2)
        ax.scatter(xyz[11:,0],xyz[11:,2],-xyz[11:,1],s=9,color="#d9913f")
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),
               zlim=(-center[1]-radius,-center[1]+radius),title=title,xlabel="Body x",ylabel="Body z",zlabel="Up (-y)")
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=15,azim=-70)
    fig.suptitle(f"{row['sample_id']}, progress frame 48/96; dimensionless torso units\nMedian cross-view disagreement; no calibrated 3D ground truth")
    output=HERE/"figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output/"fusion_example.png",dpi=160)
    plt.close(fig)
    vis=np.stack([a[1] for a in aligned])
    weights=vis**2
    empty=weights.sum(0)<1e-8
    weights[:,empty]=1
    weights/=weights.sum(0,keepdims=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for ax,joint,title in zip(axes,[15,16],["Left wrist","Right wrist"]):
        for camera in range(3):
            ax.plot(weights[camera,:,joint],label=f"Camera {camera+1}")
        ax.set(title=title,xlabel="Normalized action progress (96 frames)",ylabel="Normalized fusion weight",ylim=(0,1))
        ax.legend()
    fig.suptitle("Joint weights in the same median-disagreement repetition")
    fig.savefig(output/"fusion_weights.png",dpi=160)
    plt.close(fig)
    core.save_json(output/"fusion_example.json",{"sample_id":row["sample_id"],"fused_observation_index":i,
        "selection_rule":"median aligned_spread_mean among all 1111 repetitions; no classifier outcome used",
        "diagnostic":diagnostics[i]})
    print(row["sample_id"])


if __name__=="__main__":
    main()
