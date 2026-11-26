#!/usr/bin/env bash
# Archived pipeline outline. Configure dataset paths before running individual stages.
set -euo pipefail
if [[ "${DDMF_RUN_ARCHIVED_PIPELINE:-0}" != "1" ]]; then
  printf '%s\n' 'Review README.md and configure data paths, checkpoints, and templates first.' 'Run individual stages; set DDMF_RUN_ARCHIVED_PIPELINE=1 only after configuring this outline.'
  exit 1
fi
MedicalRecordNo="${1:?Usage: bash script.sh CASE_ID [x,y,z voxel spacing]}"
[[ "$MedicalRecordNo" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid CASE_ID'; exit 1; }
Resolution="${2:-0.15,0.15,0.15}"
mkdir -p "./output/$MedicalRecordNo"
#CBCT tooth segmentation
python code/dcm2jpg/dcmtojpg.py --input_dir "./data/$MedicalRecordNo" --output_path "./output/$MedicalRecordNo/slices"
python code/cbct-segmentation/tools/test.py code/cbct-segmentation/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py  code/cbct-segmentation/work_dirs/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_best_1:2/iter_160000.pth --show-dir code/cbct-segmentation/work_dirs/predict_png

#CBCT Alveolar bone segmentation
python code/jaw-segmentation/tools/test.py code/jaw-segmentation/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py  code/jaw-segmentation/work_dirs/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/iter_160000.pth --show-dir code/jaw-segmentation/work_dirs/predict_png_jaw
cp -R code/cbct-segmentation/work_dirs/predict_png "./output/$MedicalRecordNo/"
cp -R code/jaw-segmentation/work_dirs/predict_png_jaw "./output/$MedicalRecordNo/"

#CBCT tooth reconstruction
python code/CT3D/reconstruction/jpg_png2gz.py --input_dir ./output/$MedicalRecordNo/predict_png/ --output_dir ./output/$MedicalRecordNo/reconstruction/gz
python code/CT3D/reconstruction/itk_reconstruct.py --input_dir ./output/$MedicalRecordNo/reconstruction/gz/mask.nii.gz --output_dir ./output/$MedicalRecordNo/reconstruction/out.obj
code/CT3D/reconstruction/surface-smoothing/main ./output/$MedicalRecordNo/reconstruction/out.obj 5
python code/CT3D/reconstruction/trimesh_smoothing.py --input_dir ./output/$MedicalRecordNo/reconstruction/out.obj_denoised.obj --output_dir ./output/$MedicalRecordNo/reconstruction/out_smoothed_$MedicalRecordNo.stl --m $Resolution

#CBCT Alveolar bone reconstruction
python code/CT3D/reconstruction/jpg_png2gz.py --input_dir ./output/$MedicalRecordNo/predict_png_jaw/ --output_dir ./output/$MedicalRecordNo/reconstruction/jaw/gz
python code/CT3D/reconstruction/itk_reconstruct.py --input_dir ./output/$MedicalRecordNo/reconstruction/jaw/gz/mask.nii.gz --output_dir ./output/$MedicalRecordNo/reconstruction/jaw/out.obj
code/CT3D/reconstruction/surface-smoothing/main ./output/$MedicalRecordNo/reconstruction/jaw/out.obj 5
python code/CT3D/reconstruction/trimesh_smoothing.py --input_dir ./output/$MedicalRecordNo/reconstruction/jaw/out.obj_denoised.obj --output_dir ./output/$MedicalRecordNo/reconstruction/jaw/jaw_out_smoothed_$MedicalRecordNo.stl --m $Resolution

#IOS segmentation
python code/seg_iosnet/deployment/example.py --input_dir data/$MedicalRecordNo --output_dir output/$MedicalRecordNo/teeth_seg  --model_dir ./code/seg_iosnet/deployment/app/model.onnx

#CBCT and IOS registration and fusion
python code/PointSeg/CT_segmentation/main.py --tooth-path ./output/$MedicalRecordNo/reconstruction/out_smoothed_$MedicalRecordNo.stl
python code/PointSeg/Registration.py --tooth_scan_dir ./output/$MedicalRecordNo/teeth_seg/L --tooth_cbct ./code/PointSeg/processed_data/L_all_out_smoothed_$MedicalRecordNo.stl.ply --tooth_cbct_origin ./output/$MedicalRecordNo/reconstruction/out_smoothed_$MedicalRecordNo.stl
python code/PointSeg/Registration.py --tooth_scan_dir ./output/$MedicalRecordNo/teeth_seg/U --tooth_cbct ./code/PointSeg/processed_data/L_all_out_smoothed_$MedicalRecordNo.stl.ply --tooth_cbct_origin ./output/$MedicalRecordNo/reconstruction/out_smoothed_$MedicalRecordNo.stl



