use criterion::{black_box, criterion_group, criterion_main, Criterion};

use instant_clip_tokenizer::Tokenizer;

fn small(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    c.bench_function("small", |b| {
        b.iter(|| {
            tokenizer.tokenize_batch(
                black_box(["Hi", "How are you?", "I'm fine, thanks!"]),
                black_box(6),
            )
        })
    });
}

criterion_group!(tokenize_batch, small);
criterion_main!(tokenize_batch);
