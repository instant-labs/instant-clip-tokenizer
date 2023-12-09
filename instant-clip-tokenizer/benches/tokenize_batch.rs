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

fn large(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();
    c.bench_function("large", |b| {
        b.iter(|| tokenizer.tokenize_batch(black_box(TEXTS), black_box(77)))
    });
}

criterion_group!(tokenize_batch, small, large);
criterion_main!(tokenize_batch);

// These are the first 50 sentences from Google's Conceptual Captions dataset (see
// https://ai.google.com/research/ConceptualCaptions/).
const TEXTS: [&str; 50] = [
    "a very typical bus station",
    "sierra looked stunning in this top and this skirt while performing with person at their former university",
    "young confused girl standing in front of a wardrobe",
    "interior design of modern living room with fireplace in a new house",
    "cybernetic scene isolated on white background .",
    "gangsta rap artist attends sports team vs playoff game in the borough .",
    "the jetty : different types of plants to establish a variety of ecosystems .",
    "traditional ornamental floral paisley bandanna .",
    "# of the sports team skates against sports team during their game .",
    "by geographical feature category or in the city - a dome for every environment",
    "a flight was traveling when the animal got free on tuesday night",
    "even though agricultural conditions are not ideal for growing tobacco , there is indigenous production .",
    "us state speaks during a demonstration thursday .",
    "actor arrives for the premiere of the film",
    "celebrities start decorating for the christmas season lifestyle",
    "functions of government : 1 . form a more perfect union",
    "actor attends the premiere of season",
    "american football player on the field during joint training camp .",
    "companies have gone to court for the right to lie",
    "all shots by by person and rider shots can be found on his website .",
    "photo of a deer and wildfire",
    "high angle view of a businessman lying on a table and singing",
    "this is real fast food !",
    "safe deposit with money around it on a white background photo",
    "the giraffe before he was shot dead then autopsied in the presence of the zoo 's visitors , despite an online petition to save him signed by thousands of animal lovers",
    "dunes lay the blueprint for the back nine .",
    "portrait of a smiling woman stroking her dog lying on couch",
    "young business woman on a bench",
    "american football player looks downfield during the second half of a football game against sports team",
    "... and local people to deliver a new bridge",
    "actor arrives to the premiere",
    "funny animals of the week , animal pictures",
    "see the inspiring way this woman documented her travels on her prosthetic leg",
    "the sign promises as much as the glorious blue sky .",
    "architectural details of a bridge",
    "people tour and enjoy the public park during summer",
    "interesting 1930 's poster for a cosmetic company with stores .",
    "racecar driver steers his car during video game subject .",
    "vintage elegant floral card with frame decorated with black and white lilies on a pink background .",
    "heavy snow falls over a snow lined river .",
    "bright living room in the attic",
    "pop artist attends the 3rd annual at guest house",
    "illustration of a map , its flag and a comic balloon with a soccer ball in a not allowed signal",
    "rock artist performs on stage at awards held",
    "green sea turtle isolated on a white background 3d illustration",
    "person , was surprised by the staff",
    "red and white flag on the mast",
    "football player celebrates scoring for football team against football team in the final",
    "concept plug - in hybrid car on display",
    "a pencil drawing of a zebra and her baby .",
];
