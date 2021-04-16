Potential Training Dataset List
===========
| Name                      | Type         | Description                                          | Link                                                                    | License       |
|---------------------------|--------------|------------------------------------------------------|-------------------------------------------------------------------------|---------------|
| Equity Evaluation Corpus  | Gender       | Derived from semeval-2018 tweets                     | https://www.aclweb.org/anthology/S18-2005.pdf                           | ?             |
| RtGender                  | Gender       | 25M comments from FB, TED, Fitocracy, and Reddit     | https://nlp.stanford.edu/robvoigt/rtgender/                             | Research Only |
| Social Bias Frames        | Gender/Mixed | Annotated social media posts, aims at implications   | https://huggingface.co/datasets/social_bias_frames                      | cc-by-4.0     |
| Md Gender Bias            | Gender       | Annotated version of eight different datasets        | https://huggingface.co/datasets/md_gender_bias                          | mit           |
| Jigaw                     | Toxicity     | Labeled from Wikipedia comments                      | https://huggingface.co/datasets/jigsaw_toxicity_pred                    | cc0-1.0       |
| SOLID                     | Toxicity     | 9M tweets labeled by toxicity                        | https://sites.google.com/site/offensevalsharedtask/solid                | ?             |
| OLID                      | Toxicity     | 142k labeled tweets                                  | https://sites.google.com/site/offensevalsharedtask/olid                 | ?             |
| Veiled Toxicity Detection | Toxicity     | Potential toxicity counterpart to Social Bias Frames | https://github.com/xhan77/veiled-toxicity-detection/tree/main/resources | ?             |
| Winograd                  | Gender (?)   | Windograd scheme, note that this is not drop in      | https://huggingface.co/datasets/wino_bias                               | mit           |


Gender Datasets
=====================
| Name                    | Columns                                                                        | Description                                                                                                                                          |   |
|-------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| Equity Evalution Corpus | `"Sentence", "Gender"`                                                         | Easy mapping of sentence to gender of categorical variables of `["male", "female"]`                                                                  |   |
| Social Bias Frames      | `"sexYN", "sexReason", "sexPhrase", "targetCategory", "targetStereotype"`      | Looks for possibly gendered phrases, sexYN is if it contains a sexual reference, not towards a specific sex                                          |   |
| MdGender Bias           | "labels", "class_type"                                                         | Labeled either as gendered or non-gendered, but with additional references on who it is                                                              |   |
| rtGender                | Lots of different datasets, but "op_gender" and "responder_gender" can be used | Posts are provided with at least an "op_gender", which determines who wrote it, and sometimes "responder_gender", based on how people responde to it |   |

Overall, given that these datasets make a pretty clear categorical variable of ~['male', 'female']~, I think our classifier can be trained on the specific labels. Social Bias Frames does not make a clear distinction of which type, although it might be useful as a less granular classifer.

Inputs for models using EEC, MdGender, rtGender: `(gender_label, sentence)` where `gender_label` is one of `M/F`

rtGender can be extended based on the `op_gender` (as it's always present), and Social Bias Frames is less granular.



Toxicity Datasets
======================
| Name  | Columns | Description                                         |
|-------|---------|-----------------------------------------------------|
| Jigaw | "toxic" | Simple dataset that classifies toxic `0/1`          |
| SOLID | "label" | Dataset contains categorical variables of "NOT/OFF" |
| OLID  | "label" | Same as SOLID                                       |

Overall, all datasets do a simple binary classifier of some type of "offensive"/"not offensive", meaning that we could do a simple binary.

SOLID and Jigsaw both provide more information on the whether the comment is obscene, which could allow us to go deeper.
