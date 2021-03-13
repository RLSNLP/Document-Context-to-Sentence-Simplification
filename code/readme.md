# Instruction

This folder contains the code for the implementation of the SUC model. We refer to Li piji's code(https://github.com/lipiji/TranSummar) and are grateful for his help.

## Step 1 &nbsp; &nbsp; Requirements

You should use the command:
```
pip install -r requirement.txt
```
## Step 2 &nbsp; &nbsp; Preprocess

This folder should contain at least the training set files "context_downstairs_train.txt" and "sentence_train.txt", and the test set file "test.txt".
Please refer to "With_context.txt" for the format of "context_downstairs_train.txt" and "test.txt". The format of "sentence_train.txt" should refer to the format of "Without_context.txt".

You should use the command:
```
python Document-Prepare-Data.py
```
then:
```
python Sentence-Prepare-Data.py
```
**Attention:**

We have provided a dictionary file named ''vocab.txt''. However, for better performance, we strongly recommend that you make your own dictionary file based on the training set you are using and resize it as needed.

## Step 3 &nbsp; &nbsp; Train

You should use the command:
```
python main.py
```

## Step 4 &nbsp; &nbsp; Test
You should use the command:
```
python test.py
```

And you can see the output from "./wikipedia/result/simplified/"
