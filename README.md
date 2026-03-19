# AgentEditor: Instructable 3D Scene Editing with LLMs and Gaussian Splatting

<img src="\assets\figure1.png" alt="figure1"/>

### Installation
+ Install `Python >= 3.10`.
+ Install `torch >= 2.4`. We have tested on `torch==2.4.1+cu124`, but other versions should also work fine.
+ Clone our repo
```
https://github.com/researchcv/AgentEditor.git --recursive
```
+ Install dependencies:
```
sudo bash script.sh
```
## Datasets
In the experiments section of our paper, we primarily utilized two datasets: the 3D-OVS dataset and the LERF dataset.

The 3D-OVS dataset is accessible for download via the following link: [Download 3D-OVS Dataset](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing) .

The LERF dataset is accessible for download via the following link: [Download LERF Dataset](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing).

### Usage
+ First, you should set the API_KEY and URL for the large language model in config.yaml.
+ Second,  mkdir data,  then, Put the image in data.Then, execute the following commands:
```
python launch.py --train --data_dir data/you_scene --data_type colmap --output_dir outputs/you_scene  --iterations 30000
```

```
python launch.py --edit --data_dir data/you_scene --ply outputs/my_scene/final_point_cloud.ply --output_dir outputs/you_scene --prompt "you_prompt" --edit_operation Editing Type --edit_offset x,y,z
```
+ The dialogue function is executed via the following commands:

```
python launch.py --save  --data_dir data/you_scene  --ply outputs/you_scene/final_point_cloud.ply \--output_dir outputs/you_scene --prompt "you_prompt"
```