<p align="center">

  <h2 align="center"><strong>VisuRiddles: Fine-grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning</strong></h2>



# 👋 Introduction
Recent strides in multimodal large language models (MLLMs) have significantly advanced their performance in many reasoning tasks. However, Abstract Visual Reasoning (AVR) remains a critical challenge, primarily due to limitations in perceiving abstract graphics. To tackle this issue, we investigate the bottlenecks in current MLLMs and synthesize training data to improve their abstract visual perception. First, we propose VisuRiddles, a benchmark for AVR, featuring tasks meticulously constructed to assess models' reasoning capacities across five core dimensions and two high-level reasoning categories. Second, we introduce the Perceptual Riddle Synthesizer (PRS),  an automated framework for generating riddles with fine-grained perceptual descriptions. PRS not only generates valuable training data for abstract graphics but also provides fine-grained perceptual description, crucially allowing for supervision over intermediate reasoning stages and thereby improving both training efficacy and model interpretability. Our extensive experimental results on VisuRiddles empirically validate that fine-grained visual perception is the principal bottleneck and our synthesis framework markedly enhances the performance of contemporary MLLMs on these challenging tasks.
![OCRRreasoning](assets/ocr_reasoning.png)

# 🔥 News

- **[`0x/xx/2025`]**: Our paper is now accessible at [arXiv]().

- **[`0x/xx/2025`]**: Release the [dataset]() and evaluation script.


# 📌 Highlights


- **We introduce VisuRiddles, a multi-dimensional benchmark for abstract visual reasoning (AVR).** It systematically covers five key perceptual dimensions—numerosity, attribute, style, position, and spatial relation—as well as high-level analogical and consistency-based reasoning.

- **State-of-the-art MLLMs perform near random on AVR tasks.** Experiments reveal that the main bottleneck lies in fine-grained perceptual understanding of complex structures, rather than reasoning alone.

- **Fine-grained perceptual descriptions significantly boost model performance.** Recasting abstract graphics into perceptual terms enables MLLMs to solve AVR tasks much more accurately, underscoring the importance of perceptual ability.

- **We develop an automated synthesis framework for fully-annotated AVR samples.** This enables end-to-end supervision from perception to reasoning, facilitating more systematic model learning and generalization.



# 🔨 Evaluation

We have integrated OCR-Reasoning into the VLMEvalKit framework. For the environment configuration and the use of API, please refer to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Clone this repo and run the evaluate script. The code will automatically download images and annotations from HuggingFace.

```bash
git clone https://github.com/SCUT-DLVCLab/OCR-Reasoning
cd OCR_Reasoning
python run.py --data OCR_Reasoning --model Qwen2.5-VL-7B-Instruct --verbose
```


# 📖 Main Results
![main_results](assets/main_results.png)


# 🐳 Examples of VisuRiddles
![examples](assets/examples.jpg)


# VisuRiddles
VisuRiddles: Fine-grained Perception as a Primary Bottleneck for Multimodal Large Models in Complex Reasoning

**A Challenging Visual Benchmark for Evaluating Puzzle Reasoning Capabilities in Multimodal Large Models!**

**A Robust and Automated Data Synthesis Engine for Generating Realistic Riddles Data with Fine-grained Perception Information !**

# Resouces

[**🚀 Homepage**]() | [**🤗 Benchmark**]()

## News
On process.

# Description


