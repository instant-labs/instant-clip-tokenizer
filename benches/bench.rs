use criterion::{black_box, criterion_group, criterion_main, Criterion};

use instant_clip_tokenizer::Tokenizer;

fn short(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    let mut tokens = Vec::with_capacity(100);
    c.bench_function("short", |b| {
        b.iter(|| {
            tokens.clear();
            tokenizer.encode(black_box("Hello Båstad"), &mut tokens);
            tokens.len()
        })
    });
}

fn realistic(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    let mut tokens = Vec::with_capacity(100);
    c.bench_function("realistic", |b| {
        b.iter(|| {
            tokens.clear();
            tokenizer.encode(black_box("A person riding a motorcycle"), &mut tokens);
            tokens.len()
        })
    });
}

fn long_word(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    let mut tokens = Vec::with_capacity(100);
    c.bench_function("long word", |b| {
        b.iter(|| {
            tokens.clear();
            tokenizer.encode(
                black_box("donaudampfschifffahrtsgesellschaftskapitänsmütze"),
                &mut tokens,
            );
            tokens.len()
        })
    });
}

fn long_sentence(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    let mut tokens = Vec::with_capacity(100);
    c.bench_function("long sentence", |b| {
        b.iter(|| {
            tokens.clear();
            tokenizer.encode(black_box("in a hole in the ground there lived a hobbit not a nasty dirty wet hole filled with the ends of worms and an oozy smell nor yet a dry bare sandy hole with nothing in it to sit down on or to eat it was a hobbit hole and that means comfort"), &mut tokens);
            tokens.len()
        })
    });
}

criterion_group!(benches, short, realistic, long_word, long_sentence);
criterion_main!(benches);
