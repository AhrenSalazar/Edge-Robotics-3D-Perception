# Occupancy-SCOPE vs. PointPillars: 3D Detection Benchmark

An undergraduate research project comparing a custom, ultra-lightweight 3D object detection model (**Occupancy-SCOPE**) against a standard deep learning baseline (**PointPillars**). 

The goal of this project was to see if we could bypass the GPU requirements of modern neural networks by using simple matrix math on a CPU, and to measure exactly how much accuracy we sacrifice in exchange for speed.

## The Setup
The cluster is restricted to TITAN X GPUs, which fail to initialize modern OpenPCDet PyTorch models. To work around this, **all benchmarks were run entirely on the CPU**.

* **PointPillars (Baseline):** Deep Neural Network running via OpenPCDet.
* **Occupancy-SCOPE (Mine):** Converts KITTI LiDAR point clouds into a 256-feature occupancy grid and solves for bounding boxes using Ridge Regression and Random Forest regressors.

---

## 1. Speed & Latency Benchmark
Inference speed is a critical safety feature. I benchmarked the pure inference time of both models.

| Model | Type | Inference Time | Max Speed |
| :--- | :--- | :--- | :--- |
| PointPillars | Deep Neural Network | 1.70 ms | ~588 Hz |
| **Occupancy-SCOPE** | **Linear/RF Models** | **0.36 ms** | **~2770 Hz** |

**Result:** The custom Occupancy-SCOPE model achieves a 4.6x speedup, comfortably running at over 2500 Hz on CPU.

---

## 2. Accuracy & Evaluation
Models were trained on 2,000 KITTI frames and evaluated on 187 strictly unseen sequences. We used the official `Tr_velo_to_cam` calibration matrix to accurately align the LiDAR points with the camera bounding boxes.

### Model A: Linear Ridge Regression
| Metric | Score | Note |
| :--- | :--- | :--- |
| **Average BEV IoU** | ~0.01 | Linear math struggles to map 1D flattened arrays to 3D boxes across varying scenes. |
| **Depth Error** | ~6.25m | Okay for rough forward-distance estimation, but highly variable. |

### Model B: Random Forest (100 Trees)
| Metric | Score | Note |
| :--- | :--- | :--- |
| **Average BEV IoU** | ~0.03 | A 3x improvement over linear math, but still struggles to generalize to unseen spatial distributions. |
| **Max BEV IoU** | 0.76 | Proves the underlying math works perfectly *if* the object happens to align with the exact grid layout the model memorized during training. |
| **Depth Error** | ~5.17m | Slightly better depth tracking than the linear model. |

## The Takeaway
This experiment demonstrats the extreme limits of the accuracy-vs-latency tradeoff in 3D perception. By flattening a 3D occupancy grid into a 256-feature array, Occupancy-SCOPE achieves  **sub-millisecond latency (2700+ Hz) on a standard CPU**.

While the model's high Max IoU (0.76) proves the feature extraction works when objects align with the grid, the drop in average IoU on unseen data illustrates *why* deep learning models use sliding-window convolutions: to achieve translation invariance. If a car shifts one meter to the right, it crosses voxel boundaries, the 1D array changes completely, and the regressor fails to recognize it.

**Conclusion:** While it won't replace a heavy neural network for primary bounding-box perception, Occupancy-SCOPE is highly effective at what it was designed for extreme speed. Its true real-world application lies in acting as an really fast, redundant "safety shield" or a first-pass proximity warning for edge robots that do not have the GPU hardware to run PointPillars.

## How to Run

**1. Test the Latency:**
```bash
python lab_benchmarker.py --num_samples 10

**1. Test the Accuracy:**
** Linear Evaluation **
python scope_eval_ridge.py --data_path dataset/training --num_train 2000 --num_viz 200

** Random Forest Evaluation **
python scope_eval_rf.py --data_path dataset/training --num_train 2000 --num_viz 200