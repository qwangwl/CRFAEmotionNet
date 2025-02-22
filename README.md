## CRFAEmotionNet

<a href="./README_zh.md">中文版</a>

**Paper Title:** A Two-Stream Channel Reconstruction and Feature Attention Network for EEG Emotion Recognition

I have performed a simple refactor of the code for clearer readability and modified some basic hyperparameters for better performance.

Experiments were conducted on the DEAP dataset with both subject-independent and subject-dependent setups. The experiments were based on splitting the video stimuli of each subject into 1-second windows and then shuffling them.

Specifically, the experimental results are shown in the tables below:

**Table 1. Subject-Independent Experiment on the DEAP Dataset**

| Emotion | ACC                 |
| ------- | ------------------- |
| Valence | 0.98272$\pm$0.00124 |
| Arousal | 0.98306$\pm$0.00168 |

**Table 2. Subject-Dependent Experiment on the DEAP Dataset**

| Sub  | Arousal            | Valence            | Sub  | Arousal            | Valence            |
| ---- | ------------------ | ------------------ | ---- | ------------------ | ------------------ |
| 1    | 0.9991666666666668 | 0.9995833333333334 | 17   | 0.9833333333333332 | 0.9833333333333334 |
| 2    | 0.9858333333333332 | 0.9920833333333334 | 18   | 0.9895833333333334 | 0.9916666666666668 |
| 3    | 0.9979166666666666 | 0.9954166666666667 | 19   | 0.9954166666666667 | 0.9941666666666666 |
| 4    | 0.9816666666666667 | 0.9829166666666665 | 20   | 0.9974999999999999 | 0.9958333333333333 |
| 5    | 0.99               | 0.9858333333333335 | 21   | 0.9974999999999999 | 0.9916666666666666 |
| 6    | 0.9916666666666668 | 0.99375            | 22   | 0.99               | 0.9854166666666666 |
| 7    | 0.9962500000000001 | 0.9979166666666666 | 23   | 0.9962500000000001 | 0.99875            |
| 8    | 0.9845833333333331 | 0.9833333333333334 | 24   | 0.9966666666666667 | 0.9925             |
| 9    | 0.9912500000000002 | 0.9933333333333334 | 25   | 0.9904166666666667 | 0.9887499999999999 |
| 10   | 0.9925             | 0.9958333333333333 | 26   | 0.9783333333333333 | 0.9841666666666666 |
| 11   | 0.9879166666666667 | 0.9875             | 27   | 0.9970833333333333 | 0.9974999999999999 |
| 12   | 0.9975000000000002 | 0.9858333333333335 | 28   | 0.9891666666666665 | 0.9883333333333335 |
| 13   | 0.9958333333333332 | 0.9958333333333332 | 29   | 0.9966666666666667 | 0.9966666666666667 |
| 14   | 0.9833333333333332 | 0.9837499999999999 | 30   | 0.9933333333333334 | 0.9966666666666667 |
| 15   | 0.9958333333333333 | 0.9929166666666667 | 31   | 0.9970833333333333 | 0.9979166666666668 |
| 16   | 0.9966666666666668 | 0.9962500000000001 | 32   | 0.9929166666666667 | 0.99375            |

In addition, we performed extra experiments on the SEED dataset. Since directly using the raw signals from the SEED dataset would lead to too many parameters for the classifier, we only used DE features for a brief test and also changed the window size to 1.

| Session | ACC                |
| ------- | ------------------ |
| 1       | 0.999901787468081  |
| 2       | 0.9999410724808486 |
| 3       | 0.9999803574936162 |

When all the videos were split and used for experiments, the accuracy was ridiculously high. The main reason is that the samples from the same video were highly similar, which means that part of the test set was already seen by the model during training.

After completing these experiments, I didn’t feel like modifying the code anymore, so I’m wrapping it up here.

The next step will be focusing on cross-video or cross-subject experiments.

