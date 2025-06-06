## Purpose

We propose **GS-Net**, an unrolled ADMM-based network that integrates:

- Poisson likelihood modeling for fidelity,
- A generalized sparse prior via ReLU-based CNN,
- EM-based optimization with shrinkage thresholding,
- End-to-end learning of all hyperparameters.

GS-Net combines model-driven rigor with the flexibility of deep learning, improving both interpretability and performance.

![GS-Net Architecture](img/admm_pet.png)

---

## Results

We evaluated GS-Net on simulated brain and real whole-body PET datasets under various count levels (1%, 2%, 5%, 10%). Compared to FBP, EM-Gaussian, DeepPET, and other unrolled methods, GS-Net achieves superior reconstruction quality in both PSNR and SSIM.

![Comparison A](img/dose1_2.png)
![Comparison b](img/dose5_10.png)


## Usage

### Step 1: Train the GS-Net model
Run the following script to start training:
```bash
python scripts/train.py
python scripts/test.py
