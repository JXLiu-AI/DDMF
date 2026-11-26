# DDMF: Deep Dental Multimodal Fusion

Code for **Deep learning-enabled 3D multimodal fusion of cone-beam CT and intraoral mesh scans for clinically applicable tooth-bone reconstruction**, *Cell Press Patterns* (2023).

[Paper](https://doi.org/10.1016/j.patter.2023.100825) | [Original code archive](https://doi.org/10.5281/zenodo.8027716) | [Released sample data](https://doi.org/10.5281/zenodo.8027553) | [Models](https://huggingface.co/AlexLiu2023/DDMF)

DDMF combines CBCT segmentation, intraoral mesh segmentation, registration, and geometric fusion to reconstruct tooth-crown, root, and bone structures.

## Contents

| Directory | Component |
| --- | --- |
| `code/cbct-segmentation` | Tooth segmentation, including the archived modified MMSegmentation code |
| `code/jaw-segmentation` | Alveolar bone segmentation |
| `code/seg_iosnet` | IOS segmentation and deployment |
| `code/dcm2jpg` | DICOM slice conversion |
| `code/CT3D/reconstruction` | Volumetric reconstruction and surface smoothing source |
| `code/PointSeg` | Half-jaw separation, registration, and fusion |
| `code/seg_dgcnn/deployment/aligner.py` | Additional archived alignment implementation |
| `notes/checkpoint_embedded_*_config.py` | Configurations recovered from the archived CBCT checkpoints |

This maintained release starts from the published Zenodo v1 archive and adds recovered segmentation configurations and fusion scripts. Algorithms are retained; environment-specific paths and standalone examples containing case identifiers are cleaned for distribution. Notebook outputs, local build binaries, and unrelated ITK packaging scripts are excluded.

## Environment

The recorded CBCT training environment is Python 3.8, PyTorch 1.8.0, TorchVision 0.9.0, MMCV 1.3.8, and MMSegmentation 0.15.0 with local modifications. Use the bundled `mmseg` implementation for each segmentation component. An unmodified current MMSegmentation install is not a drop-in replacement.

The deployment modules additionally import Open3D, Trimesh, NumPy, SciPy, scikit-learn, OpenCV, Pillow, imageio, pydicom, SimpleITK, and ONNX Runtime. IOS mesh postprocessing requires native graph-cut extensions. Obtain the restricted GCO dependency from its provider and build it for the target system; see [THIRD_PARTY.md](THIRD_PARTY.md). These are recorded dependencies, not a newly validated installation lockfile.

## Models

The Hugging Face repository stores the original-format checkpoints separately from GitHub:

| File | Purpose |
| --- | --- |
| `tooth_iter_160000.pth` | CBCT tooth segmentation |
| `bone_iter_160000.pth` | CBCT alveolar bone segmentation |
| `ios_model.onnx` | IOS segmentation; place at `code/seg_iosnet/deployment/app/model.onnx` |

Download the CBCT files to the paths referenced by your selected configuration, or pass their locations explicitly to `tools/test.py`. The tooth archive path ends in `...22K_best_1:2/iter_160000.pth`; the bone path ends in `...22K/iter_160000.pth`. Portable model filenames on Hugging Face avoid the colon in the original tooth directory.

For example, after configuring the tooth dataset paths:

```bash
cd code/cbct-segmentation
python tools/test.py \
  work_dirs/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_best_1:2/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py \
  /path/to/tooth_iter_160000.pth --show-dir /path/to/tooth_predictions
```

Inspect the corresponding checkpoint-embedded configuration under `notes/` when selecting a work-directory configuration. Both archived CBCT checkpoints record iteration 160000. The bone checkpoint is not relabeled as a best-validation checkpoint.

## Pipeline

1. Convert DICOM slices with `code/dcm2jpg/dcmtojpg.py`, specifying both `--input_dir` and `--output_path`. Set the segmentation configurations to read these slices and the appropriate metadata.
2. Run tooth and bone inference using each component's `tools/test.py`.
3. Reconstruct the segmented volume and smooth surfaces using `code/CT3D/reconstruction`.
4. Segment IOS meshes with `code/seg_iosnet/deployment/example.py` and the ONNX model.
5. Separate jaws, register meshes, and fuse surfaces using `code/PointSeg`.

`script.sh` preserves the archived stage outline and is guarded against accidental execution. It is not a one-command installer or an automatically configured end-to-end pipeline. In particular, slice outputs must be connected to the dataset configuration, native smoothing/graph-cut components must be built, and input naming and coordinate conventions must match the deployment code.

## Data and templates

Patient scans, clinical results, full training datasets, and training logs are not included. Use the released sample data only under its existing terms. Standard jaw templates referenced by the alignment code (`standard/*.stl`) are not distributed in this release and must be obtained separately with appropriate permission. Template-dependent stages cannot run without them.

## Validation

Release checks cover source parsing, file integrity, and the distribution inventory. Full inference and the paper's quantitative experiments have not been rerun for this packaging update. This repository is research software, not a clinical device.

## Licenses

Existing licenses for redistributed third-party components are preserved in their directories. GCO library source is excluded because of its redistribution restrictions; see [THIRD_PARTY.md](THIRD_PARTY.md). No blanket license is applied to third-party components or datasets by this packaging update. Public availability does not replace the applicable component license or grant clinical-use permission.

## Citation

```bibtex
@article{liu2023ddmf,
  title={Deep learning-enabled 3D multimodal fusion of cone-beam CT and intraoral mesh scans for clinically applicable tooth-bone reconstruction},
  author={Liu, Jiaxiang and Hao, Jin and Lin, Hangzheng and Pan, Wei and Yang, Jianfei and Feng, Yang and Wang, Gaoang and Li, Jin and Jin, Zuolin and Zhao, Zhihe and Liu, Zuozhu},
  journal={Patterns},
  volume={4},
  number={9},
  pages={100825},
  year={2023},
  doi={10.1016/j.patter.2023.100825}
}
```
