## This is the official implementation of the paper "Multi-Agent VQA: Exploring Multi-Agent Foundation Models in Zero-Shot Visual Question Answering" in Pytorch.

This work explores the **zero-shot** capabilities of **foundation models** in Visual Question Answering (VQA) tasks. 
We propose an adaptive multi-agent system, named **Multi-Agent VQA**, to overcome the limitations of foundation models in object detection and counting by using specialized agents as tools. 

Unlike existing approaches, our study focuses on the system's performance without fine-tuning it on specific VQA datasets, making it more practical and robust in the open world. 
We present preliminary experimental results under zero-shot scenarios and highlight some failure cases, offering new directions for future research. 

## TODOs
- [ ] 1. Integrate [Google Gemini Pro Vision](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro-vision?pli=1) into our system (ongoing).
- [ ] 2. Integrate [LLaVA](https://llava-vl.github.io/), [CogVLM](https://github.com/THUDM/CogVLM), and other open-sourced large vision-language models into our system, and run inference on the full testing benchmarks of several Visual Question Answering datasets.
- [ ] 3. Integrate [YOLO-World](https://github.com/AILab-CVC/YOLO-World) as the object-detection agent into our system.
- [ ] 4. Experiment on more [Visual Question Answering datasets](https://paperswithcode.com/task/visual-question-answering)

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

TODO

## Environment
There are two options for setting up the required environment.
- Python virtual environment: Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
      python -m venv myenv
      source myenv/bin/activate
      pip install -r requirements.txt
  
- Docker: We have provided you with the [Dockerfile](Dockerfile) and the corresponding [Makefile](Makefile) for the Docker. To build the Docker image from the base image [```pytorch:2.2.0-cuda12.1-cudnn8-runtime```](https://hub.docker.com/r/pytorch/pytorch/tags), run

      make build-image
  
  To run the Docker container, modify the mount path in the [Makefile](Makefile) and then run

      make run

## Dataset
We evaluate our method on the widely adopted [VQA-v2](https://paperswithcode.com/dataset/visual-question-answering-v2-0) and [GQA](https://cs.stanford.edu/people/dorarad/gqa/) datasets for the Visual Question Answering task.
  
