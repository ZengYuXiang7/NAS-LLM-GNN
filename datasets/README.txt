This archive contains the following files:
    desktop-cpu-core-i7-7820x.pickle - Latency benchmark of desktop CPU (Intel Core i7-7820X)
    desktop-gpu-gtx-1080-ti.pickle   - Latency benchmark of Desktop GPU (NVIDIA GTX 1080 Ti)
    embedded-gpu-jetson-nono.pickle  - Latency benchmark of Embedded GPU (NVIDIA Jetson Nano)
    embeeded-tpu-edgetpu.pickle      - Latency benchmark of Embedded TPU (Google EdgeTPU)
    mobile-gpu-adreno-612.pickle     - Latency benchmark of Mobile GPU (Qualcomm Adreno 612 GPU)
    mobile-npu-snapdragon-855.pickle - Latency benchmark of Mobile NPU (Qualcomm Snapdragon 855 NPU)
    README.txt                       - This file
    supp.pdf                         - Supplementary material
    util.py                          - Python utility functions

LatBench is a latency dataset of NAS-Bench-201 models covering devices in desktop, embedded and mobile systems. It aims to provide reproducibility and comparability in hardware-aware NAS and to ameliorate the need for researchers to have access to a broad range of devices.

The latency benchmarks are stored as dictionary in the Pickle files:
    {(arch_vector): latency value in seconds}
(arch_vector) is a tuple representing an architecture in the NAS-Bench-201 search space. It can be converted to/from the arch_str format accepted by NAS-Bench-201 API using the provided utility functions.

Python utility functions:
    get_arch_vector_from_arch_str(arch_str) - Convert arch_str in NAS-Bench-201 to arch_vector in the pickle files
    get_arch_str_from_arch_vector(arch_vector) - Convert arch_vector in the pickle files to NAS-Bench-201 arch_vector
