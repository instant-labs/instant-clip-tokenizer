use std::fs::File;
use std::io::BufReader;

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "instant_clip_tokenizer")]
fn instant_clip_tokenizer_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}

/// A text tokenizer for the CLIP neural network.
#[pyclass]
struct Tokenizer {
    inner: instant_clip_tokenizer::Tokenizer,
}

#[pymethods]
impl Tokenizer {
    /// Create a new `Tokenizer` using the vocabulary data bundled with this library.
    ///
    /// The resulting `Tokenizer` is suitable for use with the original CLIP model.
    ///
    /// Note that creating a new `Tokenizer` is expensive, so it is recommended to create the
    /// `Tokenizer` once and then reuse it.
    #[new]
    fn new() -> Self {
        Tokenizer {
            inner: instant_clip_tokenizer::Tokenizer::new(),
        }
    }

    /// Create a new `Tokenizer` by reading the vocabulary data from the given filename.
    ///
    /// The data must be in the format used by the original CLIP tokenizer implementation from
    /// OpenAI.
    ///
    /// Note that creating a new `Tokenizer` is expensive, so it is recommended to create the
    /// `Tokenizer` once and then reuse it.
    #[staticmethod]
    fn load(filename: &str, max_vocabulary_size: u16) -> PyResult<Self> {
        Ok(Self {
            inner: instant_clip_tokenizer::Tokenizer::with_vocabulary(
                BufReader::new(File::open(filename)?),
                max_vocabulary_size,
            )?,
        })
    }

    /// Tokenize one or multiple input strings.
    ///
    /// Each given input string is encoded using the `encode` method and the numeric representation
    /// written to a row in the resulting two-dimensional numpy array of shape `(len(texts),
    /// context_length)`, with the special `<start_of_text>` token prepended, and `<end_of_text>`
    /// appended to each text.
    ///
    /// The individual input strings are lowercased before being tokenized, but otherwise no
    /// pre-processing is performed.
    ///
    /// `context_length` is the maximum number of tokens per each text and defaults to `77` which is
    /// the correct value for all current CLIP models. If tokenization results in less than
    /// `context_length` tokens the resulting row will be padded with trailing zeros. If tokenizing
    /// an input text results in too many tokens, the token sequence will be truncated to fit within
    /// the resulting row of length `context_length`, always including the `<start_of_text>` and
    /// `<end_of_text>` marker tokens.
    ///
    /// The resulting array can be passed directly to the CLIP neural network.
    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        input: TokenizeBatchInput,
        context_length: Option<usize>,
    ) -> PyResult<&'py PyArray2<u16>> {
        let context_length = context_length.unwrap_or(77);
        if context_length < 3 {
            return Err(PyValueError::new_err("context_length is less than 3"));
        }
        let result = match input {
            TokenizeBatchInput::Single(text) => self.inner.tokenize_batch([text], context_length),
            TokenizeBatchInput::Multiple(texts) => self.inner.tokenize_batch(texts, context_length),
        };
        Ok(result.into_pyarray(py))
    }

    /// Encode a `text` input as a sequence of tokens.
    ///
    /// The encoded token sequence does not include the special `<start_of_text>` and
    /// `<end_of_text>` marker tokens. When these are needed you can either use the `tokenize_batch`
    /// method instead, or add them manually by using the `start_of_text` and `end_of_text` methods.
    fn encode(&self, text: &str) -> Vec<u16> {
        let mut tokens = Vec::with_capacity(text.len());
        self.inner.encode(text, &mut tokens);
        tokens
            .into_iter()
            .map(instant_clip_tokenizer::Token::to_u16)
            .collect()
    }

    /// Convert a sequence of `tokens` back to a textual representation.
    ///
    /// Due to the way whitespace and lowercasing is handled a sequence of tokens will not always be
    /// decoded back to the exact same text that `encode` was called with, in other words,
    /// `decode(encode(text)) == text` does not always hold true. Hence, this function is mostly
    /// useful for debugging purposes.
    fn decode(&self, tokens: Vec<u16>) -> PyResult<String> {
        let tokens = tokens
            .into_iter()
            .map(|t| {
                instant_clip_tokenizer::Token::from_u16(t, &self.inner)
                    .ok_or_else(|| PyValueError::new_err(format!("invalid token: {t}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.inner.decode(tokens))
    }

    /// Returns the special `<start_of_text>` marker token.
    fn start_of_text(&self) -> u16 {
        self.inner.start_of_text().to_u16()
    }

    /// Returns the special `<end_of_text>` marker token.
    fn end_of_text(&self) -> u16 {
        self.inner.end_of_text().to_u16()
    }
}

#[derive(FromPyObject)]
enum TokenizeBatchInput<'a> {
    #[pyo3(transparent, annotation = "str")]
    Single(&'a str),
    #[pyo3(transparent, annotation = "list[str]")]
    Multiple(Vec<&'a str>),
}
