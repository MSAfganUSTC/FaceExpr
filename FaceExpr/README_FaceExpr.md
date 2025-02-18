# FaceExpr training example for personalized facial expression generation

FaceExpr is a method to personalize text-to-image models like Stable Diffusion with a focus on generating facial expressions, given just a few (3~5) images of a subject..

The `train_FaceExpr_afgan.py` script shows how to implement the training procedure with for FaceExper. 


This will also allow us to save the trained model parameters to the local directory.

## Running locally with PyTorch
## Cloning Stable Diffusion XL Model
To set up the Stable Diffusion XL model, follow these steps:
Navigate to the Code_Implementation directory (or your desired directory where you want to store the model weights).
```bash
cd Code_Implementation
```
Create a new directory named stable_diffusion_weights to store the model files.
```bash
mkdir stable_diffusion_weights
```
Inside the stable_diffusion_weights directory, clone the Stable Diffusion XL model from Hugging Face using the following command:
```bash
cd stable_diffusion_weights
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9

```
Once you complete these steps, the Stable Diffusion XL model will be cloned into the stable_diffusion_weights directory.


### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

Extract code_implemenataion .zip file place into the local directory

Then cd in the `Code_Implementation\FaceExpr\FaceExpr` folder and run
```bash
pip install -r requirements_FacrExpr.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.14.0` installed in your environment.

Now, we can launch training using:

```bash
PROJECT_NAME = "Stable_Diffusion_FaceExpr"
MODEL_NAME = "/home/afgan/stable_diffusion_weights/stabilityaistable-diffusion-xl-base-1.0"
DATA_DIR = "/home/afgan/Input_Identity_dataset_512x512/e_w4/"
OUTPUT_DIR = "/home/afgan/FaceExpr_Weights/ID_e_w4_fine_tune"

accelerate launch train_FaceExpr_afgan.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of e_w4 woman" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --validation_epochs=25 \
  --seed="0" \
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, be sure to install `wandb` with `pip install wandb`. Don't forget to call `wandb login <your_api_key>` before training if you haven't done it before.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

## Notes

Additionally, we welcome you to explore the following CLI arguments:

* `--lora_layers`: The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only.
* `--complex_human_instruction`: Instructions for complex human attention as shown in [here](https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55).
* `--max_sequence_length`: Maximum sequence length to use for text embeddings.


We provide several options for optimizing memory optimization:

* `--offload`: When enabled, we will offload the text encoder and VAE to CPU, when they are not used.
* `cache_latents`: When enabled, we will pre-compute the latents from the input images with the VAE and remove the VAE from memory once done.
* `--use_8bit_adam`: When enabled, we will use the 8bit version of AdamW provided by the `bitsandbytes` library.

