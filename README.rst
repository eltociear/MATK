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

To get started, run the following command::

  pip install -r requirements.txt

***************
Main Features
***************

* Provides a framework for training and evaluating a different multimodal classification models on well known hateful memes datasets
* Allows for efficient experimentation and parameter tuning through modification of configuration files (under configs directory)
* Evaluate models using different state-of-the-art evaluation metrics such as Accuracy and AUROC


***************
Examples and Tutorials
***************

Coming soon...

**************************
Datasets and Preprocessing
**************************


Supported Datasets
~~~~~~~~~~~~~~~~~~
+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset                      | References                                                                                                                                                     |
+==============================+================================================================================================================================================================+
| Facebook Hateful Memes (FHM) |  `[Paper] <https://arxiv.org/pdf/2112.04482.pdf>`_ `[Dataset] <https://www.drivendata.org/accounts/login/?next=/competitions/70/hateful-memes-phase-2/data/>`_ |
+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Fine Grained FHM             |  `[Paper] <https://aclanthology.org/2021.woah-1.21.pdf>`_  `[Dataset] <https://github.com/facebookresearch/fine_grained_hateful_memes/tree/main/data>`_        | 
+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MAMI                         |  `[Paper] <https://aclanthology.org/2022.semeval-1.74.pdf>`_ `[Dataset] <https://competitions.codalab.org/competitions/34175>`_                                | 
+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HarMeme                      |  `[Paper] <https://aclanthology.org/2021.findings-acl.246.pdf>`_ `[Dataset] <https://github.com/di-dimitrov/harmeme>`_                                         |   
+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+

Adding Custom Datasets
~~~~~~~~~~~~~~~~~~
*Dataset Format.* Each meme dataset is required to have the following fields:

* img: image filepath
* text: superimposed/overlaid text
* {labels}: the label name changes based on the dataset (i.e. hateful, offensive)


**************************
Meme Models and Evaluation
**************************
Supported Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model      | Paper                                                                                                                                                                     | 
+============+===========================================================================================================================================================================+
| BART       | `[Paper] <https://aclanthology.org/2020.acl-main.703.pdf>`_ `[Code] <https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration>`_ |
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PromptHate | `[Paper] <https://arxiv.org/pdf/2302.04156.pdf>`_ `[Code] <https://gitlab.com/bottle_shop/safe/prompthate>`_                                                              |
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Supported Vision-Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model       | References                                                                                                                                                |
+=============+===========================================================================================================================================================+
| FLAVA       | `[Paper] <https://arxiv.org/pdf/2112.04482.pdf>`_ `[Code] <https://huggingface.co/docs/transformers/model_doc/flava#transformers.FlavaModel>`_            |   
+-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| VL-T5       | `[Paper] <https://arxiv.org/pdf/2102.02779.pdf>`_ `[Code] <https://github.com/j-min/VL-T5>`_                                                              |   
+-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| LXMERT      | `[Paper] <https://arxiv.org/pdf/1908.07490.pdf>`_ `[Code] <https://huggingface.co/docs/transformers/model_doc/lxmert#transformers.LxmertModel>`_          |
+-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| VisualBERT  | `[Paper] <https://arxiv.org/pdf/1908.03557.pdf>`_ `[Code] <https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertModel>`_ |
+-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Project Status
~~~~~~~~~~~~~~
+-------------------+------+------------+-------+-------+--------+------------+
|                   | BART | PromptHate | FLAVA | VL-T5 | LXMERT | VisualBERT |
+===================+======+============+=======+=======+========+============+
| FHM               |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| Fine Grained FHM  |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| MAMI              |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| HarMeme           |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| Harm-C + Harm-P   |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| Misogynistic-MEME |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+
| MET-Meme          |      |            |       |       |        |            |
+-------------------+------+------------+-------+-------+--------+------------+

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