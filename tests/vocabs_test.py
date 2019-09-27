import os


def build_source_vocab(file_path, name):
    vocab = []
    with open(file=file_path, mode="r+", encoding="utf-8", buffering=8192) as fin:
        file_list = fin.readlines()
    for word in file_list:
        word = word.strip("\n")
        wl = word.split(" ")
        vocab.extend(wl)
    vocab_path = os.path.join(os.path.dirname(file_path), name)
    vocab = set(vocab)
    with open(file=vocab_path, mode="w+", encoding="utf-8", buffering=8192) as fout:
        fout.write("<blank>" + "\n")
        fout.write("<s>" + "\n")
        fout.write("<\s>" + "\n")
        for w in vocab:
            fout.write(w + "\n")
    return vocab_path


def merge_vocabs_to_target(input_file, input_file_2):
    vocabs = set()
    with open(input_file, mode="rt", encoding="utf8", buffering=8192) as f_src, \
            open(input_file_2, mode="rt", encoding="utf8", buffering=8192) as f_src_2:
        for line in f_src:
            words = line.strip("\n").strip()
            vocabs.add(words)
        for line in f_src_2:
            words = line.strip("\n").strip()
            vocabs.add(words)
    vocabs = sorted(vocabs)
    with open(input_file_2, mode="wt", encoding="utf8", buffering=8192) as f_tgt_2:
        for vs in vocabs:
            f_tgt_2.write(vs + "\n")


def test_collect_vocabs():
    input_file = build_source_vocab(file_path=r"E:\LTTProject\SEQ2SEQ\data\total.ocr", name="source_vocab")
    input_file_2 = build_source_vocab(file_path=r"E:\LTTProject\SEQ2SEQ\data\total.std", name="target_vocab")
    merge_vocabs_to_target(input_file, input_file_2)


if __name__ == '__main__':
    test_collect_vocabs()
