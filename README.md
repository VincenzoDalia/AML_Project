# AML 2023/2024 Project - Activation Shaping for Domain Adaptation
"Activation Shaping for Domain Adaptation" - Vincenzo Dalia, Mario Todaro, Fabio Barbieri

## Base Code Structure
In the following, you can find a brief description of the included files, alongside where it is suggested to edit the code.

| File | Description | Code to Implement |
| ---- | ----------- | ----------------- |
|[...]|
| `models/resnet.py` | contains the resnet18 baseline |
| `models/as_module.py` | contains the custom Activation Shaping Module |
| `models/ras_resnet.py` | contains the architecture that performs random shaping with the custom module |
| `models/da_resnet.py` | contains the architecture that performs domain adaptation with the custom module |

## Running The Experiments
To run the experiments you can use, the provided launch scripts. Example usage:
```
./launch_scripts/baseline.sh cartoon
./launch_scripts/random_maps.sh sketch --mask_ratio 0.9
./launch_scripts/domain_adapt.sh photo --layers 3.1.2 4.1.bn2 --topK --tk_treshold 0.5 --no_binarize
```

In the following, you can find a brief description of the relevant command line arguments when launching the experiments.

### Command Line Arguments (Hyperparameters)
| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  | Description |
| -------- | ----------- |
| `--experiment` | the identifier of the experiment to run. (Eg. `baseline`) |
| `--experiment_name` | the name of the path where to save the checkpoint and logs of the experiment. (Eg. `baseline/cartoon`) |
|[...]|
| `--layers` | The layers after which to hook the activation shaping module. Any number of strings following this pattern: RESNET_LAYER.LEVEL CONV_NUM, for example: 2.0.1 corresponds to resnet layer2.0.conv1. To hook a relu layer, use the pattern : 2.0.r, for layer2.0.relu. To hook the avgpool, use "avgpool". To hook the first convolution, resnet conv1, use : 1. Invalid/Duplicated layers are ignored. |
| `--topK` | if set, topK shaping of activation map is performed (for both experiments) |
| `--tk_treshold` | if set, and the topK flag is set, this controls the K in the topK shaping of activation maps |
| `--no_binarize` | if set, the activation shaping module will not binarize the masks |
| `--mask_ratio` | it controls the number of ones in the random masks (random_maps experiment only) |

*Note: you must use --flag running the scripts to pass other args then the experiment name. so:*
```
./launch_scripts/domain_adapt.sh cartoon --layers 2.1.2 4.1.2 --topK --no_binarize  --tk_treshold=0.5

./launch_scripts/random_maps.sh cartoon --layers 2.1.2 --mask_ratio 0.9
```

## Baseline Results (see point 0. of the project)
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| Baseline |            54.52             |             40.44           |            95.93           |  63.63  |


## Domain adaptation most improving experiments
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| 4.1.2, 4.1.bn2, topK90, eps = 0.1 |            60,58             |             49,27           |            94,73           |  68,19  |

## Random shaping most improving experiments
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| 2.1.2 |            ...             |             ...           |            ...           |  ...  |