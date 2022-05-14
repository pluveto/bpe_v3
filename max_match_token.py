

class Tokenizer:
    def __init__(self):
        self.user_dict = {}

    def add_dict(self, new_dict: dict) -> 'Tokenizer':
        """添加用户词典"""
        self._preproc(new_dict)
        return self

    def _preproc(self, new_dict: dict):
        # merge new_dict to user_dict
        for k, v in new_dict.items():
            if k in self.user_dict:
                self.user_dict[k] += v
            else:
                self.user_dict[k] = v

        self.user_dict_items = sorted(self.user_dict.items(),
                                key=lambda x: (-len(x[0]), x[1]))
        pass

    def tokenize(self, spaced_sentence: list) -> list:
        """对句子分词"""
        si = 0
        result = []
        while si < len(spaced_sentence):
            matched = False
            prevlen = 0
            to_match = ''
            for w, c in self.user_dict_items:
                if prevlen != len(w):
                    to_match = ''.join(spaced_sentence[si:si + len(w)])
                    prevlen = len(w)
                if w == to_match:
                    result.append(w)
                    si += len(w)
                    matched = True
                    break
            if not matched:
                result.append(spaced_sentence[si])
                si += 1

        return result


def load_word_freq_map(path, thold = 0):
    with open(path, 'r', encoding='utf-8') as f:
        word_freq_map = {}
        for line in f:
            word, freq = line.split('\t')
            freq = int(freq)
            if freq > thold:
                word_freq_map[word] = int(freq)
    return word_freq_map


def main():
    print(
        Tokenizer()
        .add_dict(load_word_freq_map("output/vocab_BPE.txt_1"))
        .add_dict(load_word_freq_map("output/vocab_BPE.txt_2"))
        .add_dict(load_word_freq_map("output/vocab_BPE.txt_3"))
        .add_dict(load_word_freq_map("output/vocab_BPE.txt_4"))
        .tokenize(list("他是一位声誉很高的学者，凭借丰富的知识储备，在这部重要的作品中，就弥源太是否皈依进行过讨论。"))
    )


if __name__ == "__main__":
    main()
