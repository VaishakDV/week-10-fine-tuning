# Week 10 - QLoRA Fine-Tuning

Fine-tuned Llama 3.2 (3B) on Indian geography and culture 
using QLoRA on Google Colab's free T4 GPU.

## What it does
- Creates a 51-example instruction dataset on Indian geography and culture
- Fine-tunes Llama 3.2 using QLoRA (4-bit quantization + LoRA adapters)
- Trains in under 1 minute on free T4 GPU
- Evaluates fine-tuned model on seen and unseen questions

## What I learned
- LoRA — training only 0.14% of model parameters (4.5M out of 3.2B)
- QLoRA — 4-bit quantization to fit large models on free GPU
- Instruction dataset format — ### Instruction / ### Response
- Train/test split for honest evaluation
- Loss as a training metric — lower = better
- Fine-tuning shapes response style, not just adds facts
- GPU vs CPU training speed difference (52s vs 1.5hrs)

## Training Results
- Epochs: 3
- Final training loss: 1.593
- Final validation loss: 1.562
- Training time: 52 seconds on T4 GPU

## Tech Stack
- Llama 3.2 3B (Meta, via Hugging Face)
- QLoRA (4-bit quantization + LoRA adapters)
- Google Colab T4 GPU (free)
- Libraries: transformers, peft, trl, bitsandbytes

## How to run
1. Open the notebook in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Verify torch.cuda.is_available() returns True before training
