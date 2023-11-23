# CRFAEmotionNet

### usage

main.py is our main program with several parameters including
```
--data DATA, -d DATA  data path
--emotion EMOTION, -e EMOTION
                    Which emotion needs to be trained. By default, both Valence and Arousal are trained
--stream STREAM, -s STREAM
                    Which stream needs to be trained. By default, both static and dynamic streams are trained
--mode MODE, -m MODE  train or test
```

data path is the location of the .dat file in the DEAP dataset, the default is `data\`, you can put 32 .dat files in `data\`.

The training and testing of this model are separated, Therefore you must provide the mode parameter.


For example
```python
python main.py --emotion valence --stream static --mode train
```
This will train a static stream model with valence as the target.

```python
python main.py --mode train
```
This will serially train two streams of all emotional states

When testing, data_path and stream are not mandatory parameters, because during training, the data is stored in tmp_path.

For example
```python 
python main.py --emotion valence --mode test
```
This will evaluate the model of valence emotion. It is important to note that the model must be trained before evaluation.

