import numpy as np

import instant_clip_tokenizer

def main():
    tokenizer = instant_clip_tokenizer.Tokenizer()
    
    tokens = tokenizer.tokenize_batch("blaa bla", context_length=10)
    expected = np.array([[49406, 2205, 320, 19630, 49407, 0, 0, 0, 0, 0]])
    assert np.array_equal(tokens, expected)
    print(tokens)

    tokens = tokenizer.tokenize_batch(["Hi", "How are you?", "I'm fine, thanks!"], context_length=6)
    expected = np.array([
        [49406, 1883, 49407, 0, 0, 0],
        [49406, 829, 631, 592, 286, 49407],
        [49406, 328, 880, 3797, 267, 49407],
    ])
    assert np.array_equal(tokens, expected)
    print(tokens)

    tokens = tokenizer.encode("Hello world!!!")
    assert tokens == [3306, 1002, 995]
    print(tokens)

    decoded = tokenizer.decode([320, 2533, 6765, 320, 10297])
    assert decoded == "a person riding a motorcycle "
    print(decoded)

if __name__ == '__main__':
    main()
