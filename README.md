# DeepRel

This project provides a convolutional neural network model for relation
extraction.

## Getting Started

These instructions will get you a copy of the project up and  running on your
local machine for development and testing  purposes.

### Prerequisites

1. install the required packages

```bash pip install -r requirements.txt ```

2. Follow the [instruction ](http://www.nactem.ac.uk/GENIA/tagger/) to install
`geniatagger` in `$GENIATAGGER_PATH`

3. [Download](http://nlp.stanford.edu/software/corenlp.shtml#Download) and
unpack the compressed file to `$CORENLP_PATH`.

4. [Download](http://bio.nlplab.org/) the `word2vec` model to '$WORD2VEC_PATH'

### Prepare the dataset

The program needs three separated datasets in JSON format: training,
development, and test.  Each file contains sentences with annotations and
relations. `deeprel_schema.json` describes the data format.  The folder
`examples` contains some examples.

### Prepare the configuration file

The program needs `ini` file to configure the locations of `genia tagger`,
`stanford corenlp`, etc.  An example of `ini` file can be found in `examples`.

Note: please do not include `.json` for each dataset.

## Contributing

Please read
[CONTRIBUTING](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for
details on our code of conduct, and the process for submitting pull requests to
us.

## Developers

* Yifan Peng - *Initial work- [NCBI,NLM,NHI](https://github.com/yfpeng)

See also the list of [contributors](https://github.com/ncbi-
nlp/DeepRel/contributors) who participated in this project.

## Acknowledgments

This work was supported by the Intramural Research Programs of the National
Institutes of Health, National Library of Medicine. We are also grateful to
Robert Leaman for the helpful discussion.

## Reference

* Peng Y, Lu Z. Deep learning for extracting protein-protein interactions from
* biomedical literature. In Proceedings of BioNLP workshop. 2017.
