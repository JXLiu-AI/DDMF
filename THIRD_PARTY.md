# Third-party components

- The CBCT and jaw segmentation components contain modified MMSegmentation source. Preserve their Apache-2.0 notices in each component's `LICENSE` file.
- The bundled libigl geometry source has its own MPL/GPL notices. Consult `code/CT3D/reconstruction/libigl/LICENSE.MPL2` and `LICENSE.GPL` before use or redistribution.
- **GCO 3.0 is not bundled here.** Its archived license restricts redistribution to the original author or parties with express permission. Obtain the library and required accompanying geometry headers from the original provider: <https://vision.cs.uwaterloo.ca/code/> (GCO / multi-label optimization). Restore the expected headers and sources under `code/seg_iosnet/deployment/gco-v3.0/` according to the provider's license and the includes in `smoothing.cpp`. The retained `smoothing.cpp` is the deployment wrapper from the published DDMF archive, not a replacement for the GCO library. GCO requires citations and restricts commercial use; consult the provider's current terms.

This release does not grant additional rights to third-party software, jaw templates, clinical data, or models from other providers.
