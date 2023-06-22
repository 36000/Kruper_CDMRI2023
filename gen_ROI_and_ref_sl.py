import numpy as np
import nibabel as nib
import os.path as op
import os

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from AFQ.data.fetch import read_mni_template

wmvp_path = op.join(op.expanduser('~'), 'AFQ_data', 'WMVP')
os.makedirs(wmvp_path, exist_ok=True)

right_centroid_on = np.asarray([
    [-4, -6, -22],
    [-5, -9, -24],
    [-7, -12, -26],
    [-10, -15, -28],
    [-13, -20, -30],
    [-16, -23, -32],
    [-18, -26, -34],
    [-23, -32, -35],
    [-28, -39, -38],
    [-30, -45, -40]
])
left_centroid_on = right_centroid_on.copy()
left_centroid_on[:, 0] = -left_centroid_on[:, 0]

right_centroid_ot = np.asarray([
    [-4, -6, -22],
    [-3, -4, -20],
    [-7, -1, -18],
    [-10, 2, -16],
    [-15, 6, -12],
    [-22, 12, -10],
    [-24, 17, -8]
])
left_centroid_ot = right_centroid_ot.copy()
left_centroid_ot[:, 0] = -left_centroid_ot[:, 0]

right_centroid_on[:, 0] = (-right_centroid_on[:, 0] + 96)
right_centroid_on[:, 1] = (-right_centroid_on[:, 1] + 132)
right_centroid_on[:, 2] = right_centroid_on[:, 2] + 78
left_centroid_on[:, 0] = (-left_centroid_on[:, 0] + 96)
left_centroid_on[:, 1] = (-left_centroid_on[:, 1] + 132)
left_centroid_on[:, 2] = left_centroid_on[:, 2] + 78

right_centroid_ot[:, 0] = (-right_centroid_ot[:, 0] + 96)
right_centroid_ot[:, 1] = (-right_centroid_ot[:, 1] + 132)
right_centroid_ot[:, 2] = right_centroid_ot[:, 2] + 78
left_centroid_ot[:, 0] = (-left_centroid_ot[:, 0] + 96)
left_centroid_ot[:, 1] = (-left_centroid_ot[:, 1] + 132)
left_centroid_ot[:, 2] = left_centroid_ot[:, 2] + 78

d_plane_threshold = 1
roi_folder = wmvp_path
peak_ls_on = [2, 4, 7]
peak_ls_ot = [2, 5]
for bundle_name, track in zip(["right_ON", "left_ON", "right_OT", "left_OT"], [right_centroid_on, left_centroid_on, right_centroid_ot, left_centroid_ot]):
    if "OT" in bundle_name:
        peak_ls = peak_ls_ot
        roi_rad = 6
    else:
        peak_ls = peak_ls_on
        roi_rad = 8
    for peak_idx, peak in enumerate(peak_ls):
        roi = np.zeros_like(read_mni_template().get_fdata())

        min_idx = max(0, peak-1)
        max_idx = min(9, peak+1)
        n_vec = track[min_idx] - track[max_idx]
        n_vec = n_vec/np.linalg.norm(n_vec)
        c_pt = track[peak]

        dr = np.zeros((3, 2), dtype=int)
        for dim in range(3):
            dr[dim, 0] = int(c_pt[dim] - roi_rad)
            dr[dim, 1] = int(c_pt[dim] + roi_rad)
        for ii in range(dr[0, 0], dr[0, 1]):
            for jj in range(dr[1, 0], dr[1, 1]):
                for kk in range(dr[2, 0], dr[2, 1]):
                    euc_d = c_pt - np.asarray([ii, jj, kk])
                    d_plane = np.abs(np.sum(n_vec*euc_d))
                    d_point = np.sum(euc_d**2)**0.5
                    if d_plane <= d_plane_threshold and d_point <= roi_rad:
                        roi[ii, jj, kk] = 1

        if np.sum(roi) < 1:
            raise ValueError(f"{bundle_name} ROI {peak_idx} not found")
        roi = nib.Nifti1Image(
            roi.astype(np.float32),
            read_mni_template().affine)
        roi_file = f"{roi_folder}/{bundle_name}_{peak_idx}.nii.gz"
        nib.save(roi, roi_file)

ref = read_mni_template()

right_otoc_centroid = np.concatenate(
    (right_centroid_ot[peak_ls_ot[1]:0:-1],
     right_centroid_on[:peak_ls_on[0]+1]),
    axis=0)
left_otoc_centroid = np.concatenate(
    (left_centroid_ot[peak_ls_ot[1]:0:-1],
     left_centroid_on[:peak_ls_on[0]+1]),
    axis=0)
print(right_otoc_centroid)
print(left_otoc_centroid)
save_tractogram(
    StatefulTractogram(
        [right_otoc_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "right_OTOC_curve.trk"), bbox_valid_check=False)
save_tractogram(
    StatefulTractogram(
        [left_otoc_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "left_OTOC_curve.trk"), bbox_valid_check=False)

right_ot_centroid = right_centroid_ot[peak_ls_ot[1]:peak_ls_ot[0]-1:-1]
left_ot_centroid = left_centroid_ot[peak_ls_ot[1]:peak_ls_ot[0]-1:-1]
save_tractogram(
    StatefulTractogram(
        [right_ot_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "right_OT_curve.trk"), bbox_valid_check=False)
save_tractogram(
    StatefulTractogram(
        [left_ot_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "left_OT_curve.trk"), bbox_valid_check=False)

right_pon_centroid = right_centroid_on[peak_ls_on[1]:peak_ls_on[2]+1]
left_pon_centroid = left_centroid_on[peak_ls_on[1]:peak_ls_on[2]+1]
print(right_pon_centroid)
print(left_pon_centroid)
save_tractogram(
    StatefulTractogram(
        [right_pon_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "right_pON_curve.trk"), bbox_valid_check=False)
save_tractogram(
    StatefulTractogram(
        [left_pon_centroid],
        ref,
        Space.VOX),
    op.join(wmvp_path, "left_pON_curve.trk"), bbox_valid_check=False)
