# RationAnomaly: Log Anomaly Detection with Rationality via Chain-of-Thought and Reinforcement Learning

This repository contains the implementation of **RationAnomaly**, a reinforcement learning-based approach for log anomaly detection using the VeRL (Volcano Engine Reinforcement Learning) framework. The project demonstrates how to apply RLHF (Reinforcement Learning from Human Feedback) techniques to improve large language models' performance on log analysis tasks.

## 📋 Project Overview

Logs constitute a form of evidence signaling the operational status of software systems. Automated log anomaly detection is crucial for ensuring the reliability of modern software systems. RationAnomaly leverages the VeRL framework to train language models for detecting anomalies in system logs. The approach uses reinforcement learning to fine-tune pre-trained models (e.g., Llama-2-7B-Chat) to better distinguish between normal and abnormal log entries.

### Key Features

- **Reinforcement Learning Training**: Uses GRPO (Group Relative Policy Optimization) algorithm for model training
- **Multi-Dataset Support**: Supports BGL and Spirit log datasets
- **Data Rectification**: Includes tools for correcting dataset labels
- **Real-time Monitoring**: Training progress visualization and monitoring tools
- **Model Validation**: Comprehensive evaluation framework
- **Chain-of-Thought Reasoning**: Incorporates CoT prompting for better reasoning

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10
- CUDA-capable GPUs (recommended: 8x GPUs)
- Conda/Miniconda

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Gravityless/RationAnomaly.git
cd RationAnomaly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. **Apply dataset rectification** (optional but recommended):
```bash
cd datasets
python anomaly_patch.py
```

2. **Build training datasets**:
```bash
python parquet_builder.py
```

## 🔧 Usage

### Training

Start the reinforcement learning training process:

```bash
bash model_train.sh
```

The training script supports various configuration options:
- **Model**: Llama-2-7B-Chat (default)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Datasets**: BGL and Spirit log datasets
- **Hardware**: Multi-GPU training with FSDP

### Monitoring Training Progress

Use the provided Jupyter notebook to monitor training in real-time:

```bash
jupyter notebook model_watch.ipynb
```

This will display:
- Training loss curves
- Reward progression
- Performance metrics
- Estimated training time

### Model Export

After training completion, export the trained model:

```bash
bash model_export.sh
```

### Model Validation

Evaluate the trained model on test datasets:

```bash
cd validation
python model_predict.py -m "model_num" -t "testset_num" -c "conversation_type"
python model_evaluate.py -m "model_num" -t "testset_num" -s "session_name"
```

## 📁 Project Structure

```
├── README.md                 # This file
├── model_train.sh           # Training script
├── model_export.sh          # Model export script
├── model_watch.ipynb        # Training monitoring notebook
├── datasets/                # Dataset management
│   ├── anomaly_patch.json   # Label rectification data
│   ├── anomaly_patch.py     # Dataset correction script
│   ├── parquet_builder.py   # Dataset preprocessing
│   ├── trainsets/           # Training datasets
│   └── testsets/            # Test datasets
├── validation/              # Model evaluation
│   ├── model_predict.py     # Inference script
│   ├── model_evaluate.py    # Evaluation script
│   └── model_testset_list.json # Test configuration
├── verl/utils/reward_score/RationAnomaly/ # Reward function
├── models/                  # Model storage
└── targets/                 # Training outputs and checkpoints
```

## 🎯 Reward Function Design

The RationAnomaly project implements a sophisticated multi-component reward function located in `verl/utils/reward_score/RationAnomaly/` that evaluates model responses across multiple dimensions.

### Key Features

- **Adaptive Penalty System**: Stronger penalties for minority class errors
- **Multi-modal Evaluation**: Combines structural, semantic, and fluency metrics
- **Computational Efficiency**: Uses lightweight DistilGPT-2 for perplexity calculation
- **Robust Validation**: Comprehensive error handling and format checking

## 🎯 Key Components

### 1. Training Pipeline (`model_train.sh`)
- Configures multi-GPU training environment
- Sets up GRPO algorithm parameters
- Handles distributed training with FSDP
- Supports wandb logging and monitoring

### 2. Data Processing (`datasets/`)
- **anomaly_patch.py**: Applies label corrections to BGL and Spirit datasets
- **parquet_builder.py**: Converts raw data to training-ready format
- **llama2_chat_templater.py**: Handles chat template formatting

### 3. Model Validation (`validation/`)
- **model_predict.py**: Generates predictions on test sets
- **model_evaluate.py**: Computes evaluation metrics
- Supports multiple models and datasets

### 4. Real-time Monitoring (`model_watch.ipynb`)
- Visualizes training progress
- Tracks reward curves with Gaussian filtering
- Estimates remaining training time

### 5. Reward Function (`verl/utils/reward_score/RationAnomaly/`)
- **Multi-component Reward System**: Comprehensive evaluation including format validation, answer correctness, and reasoning quality
- **Format Validation**: Ensures proper XML tag structure (`<think>...</think>` and `<answer>...</answer>`)
- **Answer Scoring**: Binary classification accuracy with class-specific penalty adjustments
- **Chain-of-Thought Evaluation**: Semantic similarity (BLEU/ROUGE), length consistency, and perplexity assessment
- **Perplexity Calculation**: Uses DistilGPT-2 for computational efficiency in reasoning quality assessment

## 📊 Datasets

The project supports two primary log datasets:

- **BGL (Blue Gene/L)**: Supercomputer logs with labeled anomalies
- **Spirit**: NASA spacecraft logs with mission-critical events

Dataset rectification is applied through `anomaly_patch.json` to improve label quality.

## ⚙️ Configuration

### Training Parameters
- **Learning Rate**: 4e-7
- **Batch Size**: 32 (train/val)
- **Max Prompt Length**: 2048
- **Max Response Length**: 1024
- **Temperature**: 0.9
- **KL Coefficient**: 0.001

### Hardware Requirements
- **Minimum**: 8x GPUs with 48GB+ VRAM each
- **Recommended**: 8x A100 or V100 GPUs
- **Memory**: 128GB+ system RAM

## 📈 Results

The trained model demonstrates improved performance on log anomaly detection tasks through:
- Enhanced reasoning capabilities via Chain-of-Thought prompting
- Better distinction between normal and abnormal log patterns
- Improved generalization across different log formats

## 🤝 Contributing

This codebase is associated with our research paper on reinforcement learning for log anomaly detection. When using this code, please cite [our work](https://arxiv.org/abs/2509.14693).
```
@misc{rationanomaly,
      title={RationAnomaly: Log Anomaly Detection with Rationality via Chain-of-Thought and Reinforcement Learning}, 
      author={Song Xu and Yilun Liu and Minggui He and Mingchen Dai and Ziang Chen and Chunguang Zhao and Jingzhou Du and Shimin Tao and Weibin Meng and Shenglin Zhang and Yongqian Sun and Boxing Chen and Daimeng Wei},
      year={2025},
      eprint={2509.14693},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.14693}, 
}
```

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [VeRL Framework](https://github.com/volcengine/verl)
- [Llama-2 Model](https://ai.meta.com/llama/)
- [GRPO Algorithm for Reinforcement Learning](https://arxiv.org/pdf/2402.03300)

## 📧 Contact

For questions and issues, please open an issue in this repository or contact the authors.

---

**Note**: This repository will be made public alongside the associated research paper publication.
