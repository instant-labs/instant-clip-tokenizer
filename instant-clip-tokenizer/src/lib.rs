//! This crate provides a text tokenizer for [OpenAI's CLIP
//! model](https://github.com/openai/CLIP).
//!
//! It is intended to be a fast replacement for the original Python-based
//! tokenizer included in the CLIP repository, aiming for 100% compatibility
//! with the original implementation. It can also be used with
//! [OpenCLIP](https://github.com/mlfoundations/open_clip) and other
//! implementations using the same tokenizer.
//!
//! # Examples
//!
//! Basic usage with the bundled vocabulary data suitable for OpenAI's CLIP
//! model (requires the `openai-vocabulary-file` [crate
//! feature](#crate-features)):
//!
//! ```
//! # use instant_clip_tokenizer::{Token, Tokenizer};
//! let tokenizer = Tokenizer::new();
//! let mut tokens = vec![tokenizer.start_of_text()];
//! tokenizer.encode("Hi there", &mut tokens);
//! tokens.push(tokenizer.end_of_text());
//! let tokens = tokens.into_iter().map(Token::to_u16).collect::<Vec<_>>();
//! assert_eq!(tokens, [49406, 1883, 997, 49407]);
//! ```
//!
//! Using a custom vocabulary file:
//!
//! ```
//! # use std::fs::File;
//! # use std::io::{self, BufReader};
//! # use instant_clip_tokenizer::{Token, Tokenizer};
//! # fn main() -> io::Result<()> {
//! let f = BufReader::new(File::open("bpe_simple_vocab_16e6.txt")?);
//! let tokenizer = Tokenizer::with_vocabulary(f, 50_000)?;
//! let mut tokens = vec![tokenizer.start_of_text()];
//! tokenizer.encode("Hi there", &mut tokens);
//! tokens.push(tokenizer.end_of_text());
//! let tokens = tokens.into_iter().map(Token::to_u16).collect::<Vec<_>>();
//! assert_eq!(tokens, [49998, 1883, 997, 49999]);
//! # Ok(())
//! # }
//! ```
//!
//! # Crate features
//!
//! This crate provides two features:
//!
//! * **ndarray** - Enables the [`ndarray`](https://docs.rs/ndarray) dependency
//!   and the `Tokenizer::tokenize_batch` method that can be used to tokenize
//!   several input strings at once, returning a matrix suitable for directly
//!   passing to the CLIP neural network.
//! * **openai-vocabulary-file** - This feature bundles the default vocabulary
//!   file used for OpenAI's CLIP model together with this crate and allows
//!   users to construct a new tokenizer simply by calling [`Tokenizer::new`].
//!   When disabled, you will need to supply your own vocabulary file and
//!   construct the tokenizer using [`Tokenizer::with_vocabulary`].
//!
//! The **openai-vocabulary-file** feature is enabled by default. To disable it
//! use `default-features = false` when specifying the dependency on this crate
//! in your `Cargo.toml`.

use std::io::{self, BufRead};

use ahash::AHashMap;
use regex::Regex;

/// A text tokenizer for the CLIP neural network.
///
/// See the [module-level documentation](index.html) for more.
pub struct Tokenizer {
    byte_to_token: Box<[Token; 256]>,
    merge_rules: AHashMap<(Token, Token), Token>,
    start_of_text: Token,
    end_of_text: Token,
    decoder: AHashMap<Token, Vec<u8>>,
    word_split: Regex,
}

impl Tokenizer {
    /// Create a new `Tokenizer` using the vocabulary data bundled with this
    /// crate.
    ///
    /// The resulting `Tokenizer` is suitable for use with the original CLIP
    /// model.
    ///
    /// Note that creating a new `Tokenizer` is expensive, so it is recommended
    /// to create the `Tokenizer` once and then reuse it.
    #[cfg(any(test, feature = "openai-vocabulary-file"))]
    pub fn new() -> Tokenizer {
        static VOCABULARY_DATA: &str = include_str!("../bpe_simple_vocab_16e6.txt");
        const MAX_VOCABULARY_SIZE: u16 = 49408;
        Tokenizer::with_vocabulary(io::Cursor::new(VOCABULARY_DATA), MAX_VOCABULARY_SIZE)
            .expect("bundled vocabulary data is valid")
    }

    /// Create a new `Tokenizer` by reading the vocabulary data from `reader`.
    ///
    /// The data must be in the format used by the original CLIP tokenizer
    /// implementation from OpenAI.
    ///
    /// Note that creating a new `Tokenizer` is expensive, so it is recommended
    /// to create the `Tokenizer` once and then reuse it.
    ///
    /// # Errors
    ///
    /// If the data format is incorrect or reading from `reader` fails, then an
    /// error is returned.
    pub fn with_vocabulary(
        reader: impl BufRead,
        max_vocabulary_size: u16,
    ) -> io::Result<Tokenizer> {
        let mut string_to_token = AHashMap::default();
        let mut byte_to_token = Box::new([Token(u16::MAX); 256]);
        let mut byte_decoder = AHashMap::default();
        let r1 = b'!'..=b'~';
        let r2 = b'\xA1'..=b'\xAC'; // "¡" to "¬"
        let r3 = b'\xAE'..=b'\xFF'; // "®" to "ÿ"
        let mut token_index = 0;
        for byte in r1.chain(r2).chain(r3) {
            let token = Token(token_index);
            byte_to_token[usize::from(byte)] = token;
            let ch = char::from(byte);
            byte_decoder.insert(ch, byte);
            // Add token and also its corresponding end-of-word token
            string_to_token.insert(format!("{ch}"), token);
            string_to_token.insert(format!("{ch}</w>"), Token(token.0 + 256));
            token_index += 1;
        }
        for (idx, (byte, token)) in byte_to_token
            .iter_mut()
            .enumerate()
            .filter(|(_, token)| **token == Token(u16::MAX))
            .enumerate()
        {
            *token = Token(token_index);
            let ch = char::from_u32(idx as u32 + 256).unwrap();
            let byte = u8::try_from(byte).unwrap();
            byte_decoder.insert(ch, byte);
            string_to_token.insert(format!("{ch}"), *token);
            string_to_token.insert(format!("{ch}</w>"), Token(token.0 + 256));
            token_index += 1;
        }

        // For every increment of `token_index` above we actually also added the
        // corresponding end-of-word token, so we have to double `token_index`
        // now in order for it to be correct again.
        token_index *= 2;

        let mut merge_rules = AHashMap::default();
        for line in reader
            .lines()
            .skip(1)
            .take((max_vocabulary_size - 512 - 2).into())
        {
            let line = line?;
            let mut parts = line.split_whitespace();
            let first = parts.next().ok_or(io::Error::new(
                io::ErrorKind::Other,
                "lines must contain 2 tokens",
            ))?;
            let second = parts.next().ok_or(io::Error::new(
                io::ErrorKind::Other,
                "lines must contain 2 tokens",
            ))?;
            let first_token = *string_to_token
                .get(first)
                .ok_or(io::Error::new(io::ErrorKind::Other, "invalid merge rule"))?;
            let second_token = *string_to_token
                .get(second)
                .ok_or(io::Error::new(io::ErrorKind::Other, "invalid merge rule"))?;

            let result_token = Token(token_index);
            merge_rules.insert((first_token, second_token), result_token);
            string_to_token.insert(format!("{first}{second}"), result_token);
            token_index += 1;
        }

        // Note that the values we store in `decoder` are not necessarily valid
        // UTF-8, so we have to use `Vec<u8>` for them.
        let decoder = string_to_token
            .into_iter()
            .map(|(string, token)| (token, string.chars().map(|ch| byte_decoder[&ch]).collect()))
            .collect();

        let word_split = Regex::new(
            r"(?x)
                # Special substrings - these each get encoded as a single marker token
                <start_of_text>|<end_of_text>|
                # Common english contractions
                's|'t|'re|'ve|'m|'ll|'d|
                # Consecutive letters, single numbers, or runs of special chars
                [\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )
        .unwrap();

        Ok(Tokenizer {
            byte_to_token,
            merge_rules,
            start_of_text: Token(token_index),
            end_of_text: Token(token_index + 1),
            decoder,
            word_split,
        })
    }

    /// Tokenize a batch of multiple input strings.
    ///
    /// Each given input string is encoded using the [`encode`] method and the
    /// numeric representation written to a row in the resulting two-dimensional
    /// matrix of shape `(texts.len(), context_length)`, with the special
    /// `<start_of_text>` token prepended, and `<end_of_text>` appended to each
    /// text.
    ///
    /// The individual input strings are lowercased before being tokenized, but
    /// otherwise no pre-processing is performed.
    ///
    /// `context_length` is the maximum number of tokens per each text and
    /// should be `77` for all current CLIP models. If tokenization results in
    /// less than `context_length` tokens the resulting row will be padded with
    /// trailing zeros. If tokenizing an input text results in too many tokens,
    /// the token sequence will be truncated to fit within the resulting row of
    /// length `context_length`, always including the `<start_of_text>` and
    /// `<end_of_text>` marker tokens.
    ///
    /// The resulting matrix can be passed directly to the CLIP neural network.
    ///
    /// [`encode`]: Tokenizer::encode
    ///
    /// # Panics
    ///
    /// Panics if `context_length < 3`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ndarray::array;
    /// # use instant_clip_tokenizer::{Token, Tokenizer};
    /// let tokenizer = Tokenizer::new();
    /// let encoded = tokenizer.tokenize_batch(["Hi", "How are you?"], 5);
    /// assert_eq!(encoded, array![
    ///     [49406, 1883, 49407, 0, 0],
    ///     [49406, 829, 631, 592, 49407],
    /// ]);
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn tokenize_batch<'a, I>(&self, texts: I, context_length: usize) -> ndarray::Array2<u16>
    where
        I: IntoIterator<Item = &'a str>,
        I::IntoIter: std::iter::ExactSizeIterator,
    {
        if context_length < 3 {
            panic!("context length must be at least 3");
        }
        let texts = texts.into_iter();
        let mut result = ndarray::Array2::zeros((texts.len(), context_length));
        let mut tokens = Vec::with_capacity(context_length);
        for (text, mut result_row) in texts.zip(result.rows_mut()) {
            tokens.clear();
            tokens.push(self.start_of_text());
            self.encode(text, &mut tokens);
            tokens.truncate(context_length - 1);
            tokens.push(self.end_of_text());
            for (token, result_element) in tokens.iter().zip(&mut result_row) {
                *result_element = token.to_u16();
            }
        }
        result
    }

    /// Encode a `text` input as a sequence of tokens.
    ///
    /// The resulting tokens are appended to `out`. `text` is lowercased before
    /// being tokenized, but otherwise no pre-processing is performed.
    ///
    /// The encoded token sequence does not include the special
    /// `<start_of_text>` and `<end_of_text>` marker tokens. When these are
    /// needed you can either use the `tokenize_batch` method instead, or add
    /// them manually by using the [`start_of_text`] and [`end_of_text`]
    /// methods, as in the example below.
    ///
    /// [`start_of_text`]: Tokenizer::start_of_text
    /// [`end_of_text`]: Tokenizer::end_of_text
    ///
    /// # Examples
    ///
    /// ```
    /// # use instant_clip_tokenizer::{Token, Tokenizer};
    /// let tokenizer = Tokenizer::new();
    /// let mut tokens = vec![tokenizer.start_of_text()];
    /// tokenizer.encode("Hi there", &mut tokens);
    /// tokens.push(tokenizer.end_of_text());
    /// let tokens = tokens.into_iter().map(Token::to_u16).collect::<Vec<_>>();
    /// assert_eq!(tokens, [49406, 1883, 997, 49407]);
    /// ```
    pub fn encode(&self, text: &str, out: &mut Vec<Token>) {
        let text = text.to_lowercase();
        out.reserve(text.as_bytes().len());
        let words = self.word_split.find_iter(&text).map(|m| m.as_str());
        for word in words {
            if word == "<start_of_text>" {
                out.push(self.start_of_text());
                continue;
            } else if word == "<end_of_text>" {
                out.push(self.end_of_text());
                continue;
            }

            let start_index = out.len();
            out.extend(
                word.as_bytes()
                    .iter()
                    .map(|b| self.byte_to_token[usize::from(*b)]),
            );
            if start_index < out.len() {
                // If we added anything, mark last character as end-of-word
                // token
                out.last_mut().unwrap().0 += 256;
            }
            self.apply_merge_rules(start_index, out);
        }
    }

    fn apply_merge_rules(&self, start_index: usize, tokens: &mut Vec<Token>) {
        loop {
            let Some(((first, second), result_token)) = tokens[start_index..]
                .windows(2)
                .map(|pair| (pair[0], pair[1]))
                .filter_map(|pair| {
                    self.merge_rules
                        .get(&pair)
                        .map(|result_token| (pair, *result_token))
                })
                .min_by_key(|&(_, result_token)| result_token)
            else {
                // No merge rules left to apply -> we're done
                break;
            };

            // Reduce all occurences of this pair to `result_token`
            let mut i = start_index;
            while i < tokens.len() - 1 {
                if tokens[i] == first && tokens[i + 1] == second {
                    tokens[i] = result_token;
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Convert a sequence of `tokens` back to a textual representation.
    ///
    /// Due to the way whitespace and lowercasing is handled a sequence of
    /// tokens will not always be decoded back to the exact same text that
    /// `encode` was called with, in other words, `decode(encode(text)) == text`
    /// does not always hold true. Hence, this function is mostly useful for
    /// debugging purposes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use instant_clip_tokenizer::Tokenizer;
    /// let tokenizer = Tokenizer::new();
    /// let mut tokens = Vec::new();
    /// tokenizer.encode("Hello world!!!", &mut tokens);
    /// let decoded = tokenizer.decode(tokens);
    /// assert_eq!(decoded, "hello world !!! ");
    /// ```
    pub fn decode(&self, tokens: impl IntoIterator<Item = Token>) -> String {
        let bytes = tokens
            .into_iter()
            .flat_map(|token| {
                if token == self.start_of_text {
                    "<start_of_text>".as_bytes()
                } else if token == self.end_of_text {
                    "<end_of_text>".as_bytes()
                } else {
                    &self.decoder[&token]
                }
            })
            .copied()
            .collect::<Vec<_>>();

        String::from_utf8_lossy(&bytes).replace("</w>", " ")
    }

    /// Returns the special `<start_of_text>` marker token.
    ///
    /// See [`encode`] for an example about how to add this token to a token
    /// sequence.
    ///
    /// [`encode`]: Tokenizer::encode
    pub fn start_of_text(&self) -> Token {
        self.start_of_text
    }

    /// Returns the special `<end_of_text>` marker token.
    ///
    /// See [`encode`] for an example about how to add this token to a token
    /// sequence.
    ///
    /// [`encode`]: Tokenizer::encode
    pub fn end_of_text(&self) -> Token {
        self.end_of_text
    }
}

#[cfg(any(test, feature = "openai-vocabulary-file"))]
impl Default for Tokenizer {
    fn default() -> Tokenizer {
        Tokenizer::new()
    }
}

/// Represents a single token.
///
/// Values of this type can only be produced by calls to methods on the
/// [`Tokenizer`] type, mainly [`Tokenizer::encode`]. To input tokens into an
/// actual neural network the [`to_u16`] method should be used.
///
/// [`to_u16`]: Token::to_u16
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Token(u16);

impl Token {
    /// Create `Token` from number, validating against the given `tokenizer`.
    pub fn from_u16(token: u16, tokenizer: &Tokenizer) -> Option<Self> {
        (token <= tokenizer.end_of_text().0).then_some(Self(token))
    }

    /// Returns the numerical representation of this `Token`.
    ///
    /// The resulting number is suitable for feeding into a neural network.
    pub fn to_u16(self) -> u16 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn tokenize_batch() {
        let tokenizer = Tokenizer::new();
        let encoded = tokenizer.tokenize_batch(["Hi", "How are you?", "I'm fine, thanks!"], 6);
        let expected = ndarray::array![
            [49406, 1883, 49407, 0, 0, 0],
            [49406, 829, 631, 592, 286, 49407],
            [49406, 328, 880, 3797, 267, 49407],
        ];
        assert_eq!(encoded, expected);
    }

    #[test]
    fn encode_special_chars() {
        let tokens = encode("hello world!!!");
        assert_eq!(tokens, [Token(3306), Token(1002), Token(995)]);
    }

    #[test]
    fn decode_special_chars() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode([Token(3306), Token(1002), Token(995)]);
        assert_eq!(decoded, "hello world !!! ");
    }

    #[test]
    fn encode_apostrophe() {
        let tokens = encode("i've seen it");
        assert_eq!(tokens, [Token(328), Token(1200), Token(2041), Token(585)]);
    }

    #[test]
    fn decode_apostrophe() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode([Token(328), Token(1200), Token(2041), Token(585)]);
        assert_eq!(decoded, "i 've seen it ");
    }

    #[test]
    fn encode_short() {
        let tokens = encode("Hello Båstad");
        assert_eq!(tokens, [Token(3306), Token(65), Token(23176), Token(16485)]);
    }

    #[test]
    fn decode_short() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode([Token(3306), Token(65), Token(23176), Token(16485)]);
        assert_eq!(decoded, "hello båstad ");
    }

    #[test]
    fn encode_realistic() {
        let tokens = encode("A person riding a motorcycle");
        assert_eq!(tokens, [320, 2533, 6765, 320, 10297].map(Token));
    }

    #[test]
    fn decode_realistic() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode([320, 2533, 6765, 320, 10297].map(Token));
        assert_eq!(decoded, "a person riding a motorcycle ");
    }

    #[test]
    fn encode_long_word() {
        let tokens = encode("donaudampfschifffahrtsgesellschaftskapitänsmütze");
        assert_eq!(
            tokens,
            [
                1067, 627, 1880, 16680, 13731, 1021, 778, 4810, 2290, 619, 10279, 45588, 83, 909,
                688, 529, 42787, 978, 6522, 83, 1298
            ]
            .map(Token)
        );
    }

    #[test]
    fn decode_long_word() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode(
            [
                1067, 627, 1880, 16680, 13731, 1021, 778, 4810, 2290, 619, 10279, 45588, 83, 909,
                688, 529, 42787, 978, 6522, 83, 1298,
            ]
            .map(Token),
        );
        assert_eq!(decoded, "donaudampfschifffahrtsgesellschaftskapitänsmütze ");
    }

    #[test]
    fn encode_start_and_end_of_text() {
        let tokens = encode("<start_of_text>Hi<start_of_text>instant labs<end_of_text>");
        assert_eq!(tokens, [49406, 1883, 49406, 10635, 12021, 49407].map(Token));
    }

    #[test]
    fn encode_start_and_end_of_text_with_special_char() {
        let tokens = encode("<start_of_text>Hi!<end_of_text>");
        // Note how the "<end_of_text>" substring is not encoded as the special
        // marker token (which would be 49407), because the word-splitting regex
        // does not split it as a separate word due to the exclamation mark
        // preceeding it. This behavior is somewhat strange, but we preserve it
        // in order to stay compatible with the original Python implementation.
        assert_eq!(
            tokens,
            [49406, 1883, 0, 283, 806, 318, 539, 318, 4160, 285].map(Token)
        );
    }

    #[test]
    fn decode_start_and_end_of_text() {
        let tokenizer = Tokenizer::new();
        let decoded = tokenizer.decode([49406, 1883, 49406, 10635, 12021, 49407].map(Token));
        assert_eq!(
            decoded,
            "<start_of_text>hi <start_of_text>instant labs <end_of_text>"
        );
    }

    fn encode(input: &str) -> Vec<Token> {
        let tokenizer = Tokenizer::new();
        let mut tokens = Vec::with_capacity(input.len());
        tokenizer.encode(input, &mut tokens);
        tokens
    }
}
