# DeepRel

This project provides a convolutional neural network model for relation
extraction. 

See [`CONTRIBUTING.md`](/CONTRIBUTING.md) before filing an issue or creating a pull request.

## Getting Started

These instructions will get you a copy of the project up and  running on your
local machine for development and testing  purposes.

### Prerequisites

0. copy the project on your local machine

```bash
git clone https://github.com/ncbi-nlp/DeepRel.git
```

1. install the required packages

```bash 
pip install -r requirements.txt
```

2. Follow the [instruction ](http://www.nactem.ac.uk/GENIA/tagger/) to install
`geniatagger` to `GENIATAGGER_PATH`

3. [Download](http://nlp.stanford.edu/software/corenlp.shtml#Download) and
unpack the compressed file to `CORENLP_PATH`.

4. [Download](http://bio.nlplab.org/) the `word2vec` model to `WORD2VEC_PATH`

### Prepare the dataset

The program needs three separated datasets in JSON format: training,
development, and test.  Each file contains sentences with annotations and
relations. `deeprel_schema.json` describes the data format.  The folder
`examples` contains some examples.

To validate the dataset format, run

```bash
jsonschema -i examples/aimed-dev.json deeprel_schema.json
```

### Prepare the configuration file

The program needs `INI_FILE` to configure the locations of `genia tagger`,
`stanford corenlp`, etc.  An example of `INI_FILE` can be found in `examples`.
It is a good practice to place the `INI_FILE` in the same folder of `model_dir`, but it is not required.

### Preparse the datasets

In most cases, run the following program will parse the datasets and create input matrices for training and testing.

```bash
python run.py -pfvmsd INI_FILE
```

The program will generate intermediate files in `model_dir` specified in the `INI_FILE`.

```
all - store parsed documents in JSON
DATASET.npz - input matrix of sentences
DATASET-sp.npz - input matrix of shortest paths between two annotations
DATASET-doc.npz - input matrix of doc2vec

vocabs.json - vocabulary
word2vec.npz - maps from words to vectors
pos.npz - maps from part-of-speeches to vectors
chunk.npz - maps from chunks to vectors
arg1_dis.npz - maps from the distances between argument1 and current word to vectors
arg2_dis.npz - maps from the distances between argument2 and current word to vectors
dependency.npz - maps from dependencies to vectors
type.npz - maps from named entities to vectors
```

You can also run the `run.py` program step by step, so you can modify and check different parts of the inputs.
For example, to check how different parsers will affect the performance, you can replace the `parse tree` in each JSON file in `all` and run `-fvmstd` to regenerate the matrices.

```
python deeprel/run.py -h

Usage:
    run.py [options] INI_FILE
    
Options:
    --log <str>  Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
    -p           preparse [default: False]
    -f           create features [default: False]
    -v           create vocabularies [default: False]
    -m           create matrix [default: False]
    -s           create shortest path matrix [default: False]
    -d           create doc2vec [default: False]
    -t           test matrix format [default: False]
```

### Train the model

```bash
python deeprel/train.py INI_FILE
```

The program will train a CNN model using the training and development sets. 
The model will be stored at `model_dir` specified in the `INI_FILE`.

### Test the model

```bash
python deeprel/test.py model_dir
```

This will print a report of results using the model and test set.

## Troubleshooting

## Contributing

Please read
[CONTRIBUTING](/CONTRIBUTING.md) for
details on our code of conduct, and the process for submitting pull requests to
us.

## License

see `LICENSE.txt`.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National
Institutes of Health, National Library of Medicine. We are also grateful to
Robert Leaman for the helpful discussion.

## Reference

* Peng Y, Lu Z. Deep learning for extracting protein-protein interactions from
  biomedical literature. In Proceedings of BioNLP workshop. 2017.
