use std::env;

use instant_clip_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::new();

    let input = env::args()
        .nth(1)
        .ok_or("need one argument with text to tokenize")?;
    println!("Input: \"{input}\"");

    let mut tokens = Vec::with_capacity(input.len());
    tokenizer.encode(&input, &mut tokens);
    let tokens_readable = tokens
        .iter()
        .map(|token| tokenizer.decode([*token]))
        .collect::<Vec<_>>();
    println!("Result: {tokens:?} <-> {tokens_readable:?}");

    let decoded = tokenizer.decode(tokens);
    println!("Decoded: \"{decoded}\"");

    Ok(())
}
