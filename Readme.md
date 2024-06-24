# ViKL

## Abstract

Mammography is the primary imaging tool for breast cancer diagnosis. Despite significant strides in applying deep learning to interpret mammography images, efforts that focus predominantly on visual features often struggle with generalization across datasets. We hypothesize that integrating additional modalities in the radiology practice, notably the linguistic features of reports and manifestation features embodying radiological insights, offers a more powerful, interpretable and generalizable representation. In this paper, we announce MVKL, the first multimodal mammography dataset encompassing multi-view images, detailed manifestations and reports. Based on this dataset, we focus on the challanging task of unsupervised pretraining and propose ViKL, a innovative framework that synergizes **Vi**sual, **K**nowledge, and **L**inguistic features. This framework relies solely on pairing information without the necessity for pathology labels, which are often challanging to acquire. ViKL employs a triple contrastive learning approach to merge linguistic and knowledge-based insights with visual data, enabling both inter-modality and intra-modality feature enhancement. 

Our research yields significant findings: 

1) Integrating reports and manifestations with unsupervised visual pretraining, ViKL substantially enhances the pathological classification and fosters multimodal interactions.
2) The multimodal features demonstrate transferability across different datasets.
3) The multimodal pretraining approach curbs miscalibrations and crafts a high-quality representation space.
  
  
The MVKL dataset and ViKL code are publicly available at here to support a broad spectrum of future research.

## Code 
The repository currently contains unarranged code of ViKL. Detailed documentation and comments will be provided after paper publication.

## Dataset
Meanwhile, our [dataset](https://ucasaccn-my.sharepoint.com/:u:/g/personal/weixin_ucas_ac_cn/Een92f7S_6BHpOu6MRdtbBwB9juzT0EsNcAIgwuh3dbcbg?e=2dxxpK) will be publicly available after paper publication. Please stay tuned. 