"""Standalone script to generate word vocabularies from monolingual corpus."""

import argparse

from opennmt import tokenizers
from seq2seq.vocab import Vocab

"""Define constant values used thoughout the project."""

PADDING_TOKEN = "<blank>"
START_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"

PADDING_ID = 0
START_OF_SENTENCE_ID = 1
END_OF_SENTENCE_ID = 2


class BuildVocabs(object):

    def __init__(self, data, from_vocab, save_vocab, from_format="default", min_frequency=1, size=0, size_multiple=1,
                 without_sequence_tokens=False):
        self.data = data
        self.from_vocab = from_vocab
        self.from_format = from_format
        self.save_vocab = save_vocab
        self.min_frequency = min_frequency
        self.size = size
        self.size_multiple = size_multiple
        self.without_sequence_tokens = without_sequence_tokens

    def build_vocabs(self):
        special_tokens = [PADDING_TOKEN]
        if not self.without_sequence_tokens:
            special_tokens.append(START_OF_SENTENCE_TOKEN)
            special_tokens.append(END_OF_SENTENCE_TOKEN)

        vocab = Vocab(special_tokens=special_tokens, from_file=self.from_vocab)
        for data_file in self.data:
            vocab.add_from_text(data_file)
        vocab = vocab.prune(max_size=self.size, min_frequency=self.min_frequency)
        vocab.serialize(self.save_vocab)


def main_2():
    from_vocab = r"E:\LTTProject\SEQ2SEQ\data\source_vocab"
    save_vocab = r"E:\LTTProject\SEQ2SEQ\data\target_vocab"
    size = 0

    builder = BuildVocabs(data=[], from_vocab=from_vocab,
                          from_format="default", save_vocab=save_vocab, min_frequency=1, size=size, size_multiple=1,
                          without_sequence_tokens=False)
    builder.build_vocabs()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", nargs="*", help="Source text file.")
    parser.add_argument("--from_vocab", default=None,
                        help="Build from a saved vocabulary (see also --from_format).")
    parser.add_argument("--from_format", default="default", choices=["default", "sentencepiece"],
                        help="The format of the saved vocabulary (see also --from_vocab).")
    parser.add_argument("--save_vocab", required=True, help="Output vocabulary file.")
    parser.add_argument("--min_frequency", type=int, default=1, help="Minimum word frequency.")
    parser.add_argument("--size", type=int, default=0,
                        help="Maximum vocabulary size. If = 0, do not limit vocabulary.")
    parser.add_argument("--size_multiple", type=int, default=1,
                        help=("Ensure that the vocabulary size + 1 is a multiple of this value "
                              "(+ 1 represents the <unk> token that will be added during the training."))
    parser.add_argument("--without_sequence_tokens", default=False, action="store_true",
                        help="If set, do not add special sequence tokens (start, end) in the vocabulary.")
    tokenizers.add_command_line_arguments(parser)

    args = parser.parse_args()

    special_tokens = [PADDING_TOKEN]
    if not args.without_sequence_tokens:
        special_tokens.append(START_OF_SENTENCE_TOKEN)
        special_tokens.append(END_OF_SENTENCE_TOKEN)

    vocab = Vocab(special_tokens=special_tokens, from_file=args.from_vocab)
    for data_file in args.data:
        vocab.add_from_text(data_file)
    vocab = vocab.prune(max_size=args.size, min_frequency=args.min_frequency)
    vocab.serialize(args.save_vocab)


if __name__ == "__main__":
    main_2()
