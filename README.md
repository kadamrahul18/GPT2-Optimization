# Optimizing GPT-2 Training and Inference: Hybrid Parallelism with FlashAttention 

## **Overview**
This project aims to enhance the efficiency of GPT-2's training and inference processes by leveraging cutting-edge techniques such as:
- **Hybrid Parallelism**: Utilizing Megatron-LM to implement both data and model parallelism for efficient multi-GPU training.
- **FlashAttention**: Replacing the standard attention mechanism with a memory-efficient variant to optimize training on long-context sequences.

By integrating these techniques, we aim to achieve:
- A **30–50% reduction in training time**.
- Up to **4x improvement in inference speed**.
- Enhanced scalability for multi-GPU setups.

---

## **Key Features**
1. **Hybrid Parallelism**:
   - Combines data and model parallelism to distribute computational workloads across multiple GPUs.
   - Scales GPT-2 efficiently for limited memory setups using Megatron-LM.

2. **FlashAttention**:
   - Reduces memory overhead in attention mechanisms for long-sequence training.
   - Accelerates training while enabling longer context handling.


---

## **Challenges Addressed**
- Efficient GPU communication using NCCL for hybrid parallelism.
- Managing memory bottlenecks during long-context sequence processing.
- Balancing speed and accuracy in quantized models.
- Ensuring compatibility across the optimization stack.

---

## **Project Goals**
- **Reduced Training Time**: Minimize training duration by optimizing GPU utilization.
- **Improved Memory Efficiency**: Enable processing of longer sequences without exceeding memory limits.
- **Faster Inference**: Achieve real-time inference with quantized models.
- **Scalability**: Demonstrate effectiveness on varying GPU configurations (e.g., 4 to 8 GPUs).

---

## Contributors
- Rahul Kadam ([rsk8552@nyu.edu](mailto:rsk8552@nyu.edu))
- Varijaksh Katti ([vvk4812@nyu.edu](mailto:vvk4812@nyu.edu))
