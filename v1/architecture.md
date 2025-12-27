# Tiny Slot-BERT Architecture & Process

## 1. High-Level Architecture
**Tiny Slot-BERT** is an object-centric video understanding model designed to be lightweight and efficient. It decomposes video frames into a set of discrete "slots," where each slot represents an object or background element. These slots track objects over time and can be used to reconstruct the original video, ensuring they capture meaningful visual information.

The architecture consists of four main components:
1.  **Visual Encoder**: Extracts feature maps from input frames.
2.  **Slot Attention**: Iteratively refines a set of initialized memory slots to bind to specific objects in the features.
3.  **Temporal Transformer**: (Disabled in Phase 1) Processes slots across time to handle dynamics.
4.  **Slot Decoder**: Reconstructs the image and segmentation masks from the slots.

> **Note:** In this implementation (Phase 1), the **Temporal Transformer is disabled**. The model processes each frame independently to first learn robust object discovery (slot attention) and reconstruction before adding the complexity of cross-frame reasoning. This staged approach makes training more stable.

---

## 2. Component Details

### A. Simple Convolutional Encoder (`models/encoder.py`)
A lightweight CNN encoder is used for efficiency in Phase 1.
*   **Structure**: A stack of 4 convolutional blocks with BatchNorm and ReLU.
*   **Input**: RGB Video Frames $(B \times 3 \times H \times W)$.
*   **Downsampling**: Reduces input spatial dimensions by **16x** (e.g., $128 \times 128 \to 8 \times 8$).
*   **Output**: Feature map $(B \times N \times D)$, where $N = 64$ and $D = 384$.

### B. Slot Attention (`models/slot_attention.py`)
The core mechanism for object discovery.
*   **Input**: Features from the encoder + Positional Embeddings.
*   **Slots**: A set of $K$ learnable vectors (randomly initialized per frame with high variance to prevent collapse).
*   **Process**:
    *   **Iterative Attention**: Over $T$ iterations (default 3), slots compete to "explain" parts of the feature map via Softmax attention.
    *   **Key/Query/Value**: Slots act as Queries; Features act as Keys and Values.
    *   **Update**: Slots are updated using a GRU (Gated Recurrent Unit) based on the accumulated weighted values they attend to.
*   **Output**: Refined Slots $(B \times K \times D_{slot})$ and Attention Masks (which serve as segmentation masks).

### C. Slot Decoder (`models/decoder.py`)
Responsible for reconstructing the image from the abstract slots.
*   **Type**: **Spatial Broadcast Decoder**.
*   **Process**:
    1.  **Broadcast**: Each slot is replicated across a spatial grid.
    2.  **Positional Encoding**: Learnable 2D positional embeddings are added.
    3.  **Pixel Decoder**: A small CNN processes grids to produce RGB + Mask.
    4.  **Composition**: Final image is a weighted sum of RGB sprites using Masks.
*   **Key Feature**: The **Slot Attention Masks** are passed directly to reconstruction to force the attention mechanism to learn accurate segmentation.

---

## 3. Training Losses

The model is trained with a combination of losses designed to encourage meaningful object segmentation:

| Loss | Description | Weight |
|---|---|---|
| **Reconstruction (MSE)** | Pixel-space L2 loss between input and reconstruction. The primary signal. | 1.0 |
| **Orthogonality** | Penalizes slot similarity (cosine) to encourage diverse representations. | 0.0 (disabled) |
| **Mask Entropy** | Encourages sharp, decisive segmentation masks (low per-pixel entropy). | 0.0 (disabled) |
| **Slot Coverage** | Prevents "dead" slots that cover no area. Uses exponential penalty + entropy bonus. | 0.0 (disabled) |
| **Appearance Consistency** | **Critical.** Penalizes color variance within a slot's mask. Forces slots to align with object boundaries. | 0.0 (disabled) |

> **Note on Disabled Losses:** In certain training configurations, auxiliary losses (ortho, entropy, coverage, appearance) are set to 0 to initially let the model learn reconstruction freely. These can be progressively enabled to refine slot specialization.

---

## 4. Inference Process

1.  **Input**: A video clip (e.g., 24 frames of $128 \times 128$).
2.  **Encoding**: The encoder processes each frame independently to extract feature maps. Fourier positional embeddings are added.
3.  **Slot Extraction**: Slot Attention runs on each frame's features to extract $K$ slots (e.g., 7 slots).
4.  **Decoding**:
    *   Slots are passed to the Decoder.
    *   The decoder produces an RGB reconstruction for each slot.
    *   Per-slot images are combined into a full frame reconstruction.
5.  **Visualization**:
    *   **Original**: Input video.
    *   **Segmentation**: An argmax visualization showing which slot "owns" each pixel.
    *   **Slots**: Visualization of what individual slots "see" (original image masked by slot attention).
    *   **Reconstruction**: Model's attempt to recreate the video.
