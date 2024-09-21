## [CVPR 2024 CVinW] This is the official implementation of the paper ["Multi-Agent VQA: Exploring Multi-Agent Foundation Models in Zero-Shot Visual Question Answering"](https://arxiv.org/abs/2403.14783) in Pytorch.

[![Arxiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2403.14783)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Cite_Our_Paper-4085F4)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&q=Multi-Agent+VQA%3A+Exploring+Multi-Agent+Foundation+Models+in+Zero-Shot+Visual+Question+Answering&btnG=#d=gs_cit&t=1724265977916&u=%2Fscholar%3Fq%3Dinfo%3AHY9HN86PLnEJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)
[![Workshop](https://img.shields.io/badge/CVPR_2024-CVinW_Workshop_Spotlight-97CA00)](https://computer-vision-in-the-wild.github.io/cvpr-2024/)

***Key idea: What if a large foundation model fails at VQA? Shall we finetune it on our VQA dataset or object detection dataset? No, we should use tools, and tools are experts in their fields.***

This work explores the **zero-shot** capabilities of **foundation models** in Visual Question Answering (VQA) tasks. 
We propose an adaptive multi-agent system, named **Multi-Agent VQA**, to overcome the limitations of foundation models in object detection and counting by using specialized agents as tools. 

Existing approaches heavily rely on fine-tuning their models on specific VQA datasets with a vocabulary of size 3k. Our study instead focuses on the system's performance without fine-tuning it on specific VQA datasets, making it more practical and robust in the open world. 
We present preliminary experimental results under zero-shot scenarios and highlight some failure cases, offering new directions for future research. A full paper will be released soon.

<p align="center">
<img src=pipeline.png />
</p>

## Disclaimer
In this README, you will find instructions on all the available functionalities mentioned in the paper and they should work well. However, please understand that this repository is under development, and we currently only support GPT-4V and Gemini Pro Vision as our large vision-language models. Although you can find codes for other models or functionalities in this repository, they are either incomplete or haven't been thoroughly tested yet. Feel free to submit an issue.

## TODOs
- [x] 1. Integrate [Google Gemini Pro Vision](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro-vision?pli=1) into our system.
- [x] 2. Integrate [LLaVA](https://llava-vl.github.io/) and other open-sourced large vision-language models into our system, and run inference on the full testing benchmarks of several Visual Question Answering datasets. (ongoing)
- [ ] 3. Explore other tools available. For example, we could use [YOLO-World](https://github.com/AILab-CVC/YOLO-World) as the object-detection agent in our system.
- [ ] 4. Experiment on more [Visual Question Answering datasets](https://paperswithcode.com/task/visual-question-answering)
- [ ] 5. Release synthetic dataset and its automatic generation script.
- [ ] 6. Release a more comprehensive zero-shot VQA benchmark in the open world, including comparisons with more recent VQA works.
- [ ] 7. Release the full version of the paper.

## Citation
If you believe our work has inspired your research, please kindly cite our work. Thank you!

      @inproceedings{jiang2024multi,
        title={Multi-Agent VQA: Exploring Multi-Agent Foundation Models in Zero-Shot Visual Question Answering},
        author={Jiang, Bowen and Zhuang, Zhijun and Shivakumar, Shreyas S and Roth, Dan and Taylor, Camillo J},
        booktitle={arXiv preprint arXiv:2403.14783},
        year={2024}
      }

## Environment
There are two options for setting up the required environment.
- Docker (recommended): We have provided you with the [Dockerfile](Dockerfile) and the corresponding [Makefile](Makefile) for the Docker. To build the Docker image from the base image [```pytorch:2.2.0-cuda12.1-cudnn8-runtime```](https://hub.docker.com/r/pytorch/pytorch/tags), run

      make build-image
  
  To run the Docker container, modify the mount path in the [Makefile](Makefile) and then run

      make run

- Python virtual environment: Please check [requirements.txt](requirements.txt). You can run the following commands to create a virtual environment and install all the requirements:
    
      python -m venv myenv
      source myenv/bin/activate
      pip install -r requirements.txt

## Dataset
Due to the costs and time requirements of GPT-4V API,  we have to use a subset of the data to evaluate the performance. The test set of VQA-v2 is not publicly available and requires exact matches of the answers, making open-world answers and LLM-based graders inapplicable. We instead adopt the VQA-v2 rest-val dataset, the validation dataset in [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) and [VLMo](https://github.com/bowen-upenn/unilm/tree/master/vlmo) that was never used for training. It contains 5228 unique image-question pairs. For GQA, we take the same 1000 validation samples used in [ELEGANT](https://arxiv.org/pdf/2310.01356.pdf) for testing.

- To evaluate our method on the [VQA-v2](https://paperswithcode.com/dataset/visual-question-answering-v2-0) dataset, please follow BEiT-3's [instruction](https://github.com/microsoft/unilm/blob/master/beit3/get_started/get_started_for_vqav2.md) to download and prepare the data.

According to the instruction, you need to modify the source codes and generate the index JSON files for the dataset, so we provided the modified codes in this forked [repository](https://github.com/bowen-upenn/unilm/tree/master). Make sure you can get the file ``vqa.rest_val.jsonl``.

Our codes accept the data formats in ```v2_OpenEnded_mscoco_train2014_questions.json``` (the question file) and ```v2_mscoco_train2014_annotations``` (the annotation file), so we provide the code [utils_func/find_matched_rest_val.py](utils_func/find_matched_rest_val.py) to convert ``vqa.rest_val.jsonl`` into [```v2_OpenEnded_mscoco_rest_val2014_questions```](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [```v2_mscoco_rest_val2014_annotations.json```](https://drive.google.com/file/d/1t0-Plgv6b65L1LfVHP62iwrULorgEn4U/view?usp=sharing). **You can also download them directly by clicking on their names here.**

You should organize the dataset at the end as the following structure, but we are not going to use any training or testing splits.
```
datasets/
    coco/
        train2014/            
            COCO_train2014_000000000009.jpg                
            ...
        val2014/              
            COCO_val2014_000000000042.jpg
            ...  
        test2015/              
            COCO_test2015_000000000001.jpg
            ...
        answer2label.txt
        vqa.train.jsonl
        vqa.val.jsonl
        vqa.trainable_val.jsonl
        vqa.rest_val.jsonl
        vqa.test.jsonl
        vqa.test-dev.jsonl      
        vqa/
            v2_OpenEnded_mscoco_train2014_questions.json
            v2_OpenEnded_mscoco_val2014_questions.json
            v2_OpenEnded_mscoco_test2015_questions.json
            v2_OpenEnded_mscoco_test-dev2015_questions.json
            v2_OpenEnded_mscoco_rest_val2014_questions
            v2_mscoco_train2014_annotations.json
            v2_mscoco_val2014_annotations.json
            v2_mscoco_rest_val2014_annotations.json
```

Like what we did in our [config.yaml](config.yaml), you can add a soft link to your own ```datasets/``` folder 

    cd ~/tmp
    ln -s /path/to/your/datasets/ .
        
Otherwise, please remove the /tmp/ header from all paths in the provided [config.yaml](config.yaml).
  
- To evaluate our method on the [GQA](https://cs.stanford.edu/people/dorarad/gqa/) dataset, download the [images](https://cs.stanford.edu/people/dorarad/gqa/download.html) and the annotation file [```gqasubset1000.json```](https://drive.google.com/file/d/1SAOrdtjuYqBmY8OpUILMsaggQutaA-lE/view?usp=sharing). Again, we take the same 1000 validation samples used in [ELEGANT](https://arxiv.org/pdf/2310.01356.pdf) for a fair comparison.

You should organize the dataset at the end as the following structure.
```
datasets/
    gqa/
        images/
            1000.jpg
            ...
        gqasubset1000.json
```

## Quick Start
- Step 1. Follow instructions on [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/fe24c033820adffff66ac0eb191828542e8afe5e) to clone the repository, download the pretrained weights ```groundingdino_swint_ogc.pth```, and put it under the path ```Grounded-Segment-Anything/```. Our [Dockerfile](Dockerfile) and [Makefile](Makefile) are inherited from theirs, so there is no need to install the Grounded SAM again.

- Step 2. Follow instructions on [CLIP-Count](https://github.com/songrise/CLIP-Count/tree/43b496978e281bfae8d2c5b4b691c3910fe70a7c) to clone the repository, download the pretrained weights, rename it as ```clipcount_pretrained.ckpt```, and put it under the path ```CLIP_Count/ckpt/```. Our Dockerfile should have already taken into account its requirements.

- Step 3. Follow instructions on [OpenAI](https://platform.openai.com/docs/quickstart?context=python) to set up your OpenAI API, add a ```openai_key.txt``` file to your top directory, and paste your [API key](https://platform.openai.com/api-keys) into your txt file.
  
- Step 4. We allow command-line argparser for the following arguments:
    - ```--vlm_model``` to select the VLM for inference: ```gpt4``` or ```gemini```.
    - ```--dataset``` to select the dataset: ```gqa``` or ```vqa-v2```.
    - ```--split``` to select the dataset split: ```val-subset``` for GQA or ```rest-val``` for VQA-v2.
    - ```--verbose``` to print detailed data information and model responses during the inference.
 
  For example, you can run 

      python main.py --vlm_model gpt4 --dataset vqa-v2 --split rest-val --verbose

  in the command line to start the inference code. All the other hyper-parameters can be set at [config.yaml](config.yaml). Results will be saved under [outputs/](outputs/)

