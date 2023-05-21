MATK: Meme Analysis ToolKit
===========================

MATK (Meme Analysis Toolkit) aims at training, analyzing and comparing
the state-of-the-art Vision Language Models on the various downstream
memes tasks (i.e. hateful memes classification, attacked group
classification, hateful memes explanation generation).

.. contents:: Table of Contents 
   :depth: 2

***************
Installation
***************

Coming soon...

***************
Main Features
***************

Coming soon...

***************
Examples and Tutorials
***************

Coming soon...

**************************
Datasets and Preprocessing
**************************


Supported Datasets
~~~~~~~~~~~~~~~~~~
.. csv-table:: Supported Datasets
   :file: dataset_table.csv
   :widths: 30, 70
   :header-rows: 1


Adding Custom Datasets
~~~~~~~~~~~~~~~~~~
*Dataset Format.* Each meme dataset is required to have the following fields:

* img: image filepath
* text: superimposed/overlaid text
* {labels}: the label name changes based on the dataset (i.e. hateful, offensive)


**************************
Vision-Language Meme Models and Evaluation
**************************

Supported Models
~~~~~~~~~~~~~~~~
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Model                        | Paper                                          | Source                                                                                             |
+==============================+================================================+====================================================================================================+
| FlavaModel                   |  https://arxiv.org/pdf/2112.04482.pdf          | https://huggingface.co/docs/transformers/model_doc/flava#transformers.FlavaModel                   |
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+
| VL-T5                        | https://arxiv.org/pdf/2102.02779.pdf           | https://github.com/j-min/VL-T5                                                                     |
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+
| LxmertModel                  | https://arxiv.org/pdf/1908.07490.pdf           | https://huggingface.co/docs/transformers/model_doc/lxmert#transformers.LxmertModel                 |
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+
| BartForConditionalGeneration | https://aclanthology.org/2020.acl-main.703.pdf | https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration  |
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+
| VisualBertModel              | https://arxiv.org/pdf/1908.03557.pdf           | https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertModel        |
+------------------------------+------------------------------------------------+----------------------------------------------------------------------------------------------------+


**************************
Meme Models Analysis
**************************


**************************
Authors and acknowledgment
**************************

*  Ming Shan HEE, Singapore University of Technology and Design (SUTD)
*  Aditi KUMARESAN, Singapore University of Technology and Design (SUTD)
*  Nirmalendu PRAKASH, Singapore University of Technology and Design (SUTD)
*  Rui CAO, Singapore Management University (SMU)
*  Prof. Roy Ka-Wei LEE, Singapore University of Technology and Design (SUTD)

**************************
License
**************************

Coming soon...

**************************
Project status
**************************
[] Dataset Preprocessing
[] README.rst updates
[] Implementing analysis code